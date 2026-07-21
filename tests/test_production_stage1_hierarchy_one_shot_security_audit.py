from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oci.inference import production_stage1_hierarchy_handoff as handoff_module
from oci.inference.all_evidence_fusion_runner import (
    AllEvidenceFusionRunner,
    AllEvidenceFusionRunnerConfig,
    PreparedHierarchicalDiscoveryBatch,
    PreparedHierarchicalDiscoveryFold,
    QueryEvidenceArtifact,
    TfidfOrphanNgramArtifact,
    _current_production_hierarchy_runtime_binding,
    _issue_prepared_hierarchy_capability,
)
from oci.inference.approved_hierarchical_discovery_batch import (
    ApprovedHierarchicalDiscoveryBatchCoordinator,
    ApprovedHierarchicalDiscoveryBatchPrecommit,
)
from oci.inference.production_stage1_hierarchy_handoff import (
    AuthenticatedProductionStage1HierarchyHandoff,
    internal_hierarchy_execution_authorization,
    prepare_internal_hierarchy_execution_capability,
    run_internal_production_stage1_hierarchy_one_shot,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wrapper(path: Path, *, schema: str, body: dict) -> str:
    digest = _sha(body)
    path.write_text(
        json.dumps(
            {"schema_version": schema, "content_sha256": digest, "body": body},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return digest


class _IdentityObject:
    def __init__(self, name: str) -> None:
        self.name = name
        self.revision = 1

    def identity(self):
        return {
            "schema_version": "one_shot_security_audit_identity_v1",
            "name": self.name,
            "revision": self.revision,
        }


class _HierarchyRunner(_IdentityObject):
    pass


class _HierarchyConfig:
    def __init__(self) -> None:
        self._body = {
            "schema_version": "one_shot_security_audit_hierarchy_config_v1",
            "max_integrated_features": 8,
        }

    def as_dict(self):
        return dict(self._body)


class _ReviewPolicy:
    def __init__(self) -> None:
        self._body = {
            "schema_version": "one_shot_security_audit_review_policy_v1",
            "accepted_support_only": True,
        }

    def validate_authentication(self) -> None:
        return None

    def as_dict(self):
        return dict(self._body)


class _Provider(_IdentityObject):
    def __init__(self) -> None:
        super().__init__("production-stage1-hierarchy-provider")
        self.schedule = SimpleNamespace(partitions_by_outer_fold={1: (), 2: ()})

    def identity(self):
        body = super().identity()
        return {**body, "identity_sha256": _sha(body)}


class _Inputs:
    def __init__(self, *, dataset_sha256: str, hierarchy_config: dict) -> None:
        contract_body = {
            "schema_version": "one_shot_security_audit_contract_v1",
            "hierarchical_discovery_config": hierarchy_config,
        }
        self.hierarchical_discovery_contract_identity = {
            **contract_body,
            "content_sha256": _sha(contract_body),
        }
        self._request = {"dataset": {"sha256": dataset_sha256}}

    def _authenticated_registered_json(self, name: str):
        assert name == "immutable_build_request"
        return json.loads(json.dumps(self._request))

    def as_dict(self):
        return {
            "schema_version": "one_shot_security_audit_inputs_v1",
            "hierarchical_discovery_contract_identity_sha256": (
                self.hierarchical_discovery_contract_identity["content_sha256"]
            ),
        }


def _write_artifact(path: Path, label: str) -> Path:
    path.write_text(json.dumps({"artifact": label}) + "\n", encoding="utf-8")
    return path


def _case(
    tmp_path: Path,
    *,
    issue_prepared: bool = True,
    input_mutation=None,
):
    root = tmp_path / "preparation"
    root.mkdir()
    dataset = _write_artifact(root / "cohort.parquet", "dataset")
    legacy = _write_artifact(root / "legacy.jsonl", "legacy")
    tfidf = _write_artifact(root / "tfidf.jsonl", "tfidf")
    primary = _write_artifact(root / "primary.parquet", "primary")
    candidate = _write_artifact(root / "candidate.json", "candidate")
    query = _write_artifact(root / "query.json", "query")
    orphan = _write_artifact(root / "orphan.json", "orphan")

    hierarchy_config = _HierarchyConfig()
    review_policy = _ReviewPolicy()
    provider = _Provider()
    inputs = _Inputs(
        dataset_sha256=_file_sha(dataset),
        hierarchy_config=hierarchy_config.as_dict(),
    )
    handoff = AuthenticatedProductionStage1HierarchyHandoff(
        inputs=inputs,
        provider=provider,
    )

    gate_provider = _IdentityObject("shared-gate-provider")
    runner = object.__new__(AllEvidenceFusionRunner)
    runner.dataset_path = dataset
    runner.legacy_handoff_path = legacy
    runner.tfidf_handoff_path = tfidf
    runner.output_dir = tmp_path / "output"
    runner.hierarchical_preparation_dir = root
    runner.hierarchical_discovery_job_cache_root = root / "jobs"
    runner.hierarchical_discovery_enabled = True
    runner.hierarchical_discovery_approved_batch_sha256 = None
    runner.review_spent_evidence_provider = provider
    runner.review_partition_provider = provider
    runner.review_gate_source_provider = gate_provider
    runner.review_gate_feature_bank_provider = gate_provider
    runner.final_upstream_producer = None
    runner.raw_final_upstream_producer = None
    runner.final_causal_forest_backend = None
    runner.cache_overlay = None
    runner.hierarchical_discovery_runner = _HierarchyRunner("hierarchy-runner")
    runner.hierarchical_discovery_config = hierarchy_config
    runner.hierarchical_review_evidence_policy = review_policy
    runner.config = AllEvidenceFusionRunnerConfig()
    runner.hierarchical_max_atoms_per_chunk = 11
    runner.hierarchical_max_bytes_per_chunk = 12_000
    runner.hierarchical_max_semantic_member_ids_per_chunk = 13
    runner.legacy_primary_predictions_path = primary
    runner.candidate_pool_paths = {1: candidate}
    runner.query_evidence_by_fold = {
        1: QueryEvidenceArtifact(
            path=query,
            outer_fold=1,
            artifact_sha256=_file_sha(query),
            fit_row_fingerprint="fit-fingerprint",
            heldout_row_fingerprint="heldout-fingerprint",
        )
    }
    runner.tfidf_orphan_artifacts_by_fold = {
        1: TfidfOrphanNgramArtifact(path=orphan, artifact_sha256=_file_sha(orphan))
    }
    runner.coordinate_preserving_nuisance_view_names = None
    runner.fusion_agent = None
    runner.extraction_provider = object()
    runner.review_agent = None
    runner.tfidf_validator = None

    runtime_binding, _runtime_objects = _current_production_hierarchy_runtime_binding(runner)
    runtime = runtime_binding["body"]
    input_body = {
        "dataset": runtime["dataset_artifact"],
        "legacy_handoff": runtime["legacy_handoff_artifact"],
        "tfidf_handoff": runtime["tfidf_handoff_artifact"],
        "outer_folds": [{"outer_fold": 1}, {"outer_fold": 2}],
        **{
            key: runtime[key]
            for key in (
                "effective_runner_config",
                "extraction_cache_overlay",
                "final_causal_forest_backend",
                "final_upstream_producer",
                "frozen_review_evidence_policy",
                "hierarchical_architecture_chunk_limits",
                "hierarchical_discovery_config",
                "hierarchical_runner_identity",
                "raw_final_upstream_producer",
                "shared_first_gate_provider",
                "spent_evidence_provider",
            )
        },
    }
    if input_mutation is not None:
        input_mutation(input_body)
    input_path = root / "input.json"
    input_sha256 = _write_wrapper(
        input_path,
        schema="hierarchical_all_evidence_runner_preparation_input_v2",
        body=input_body,
    )
    packet = {"input_manifest_sha256": input_sha256, "ordered_outer_folds": [1, 2]}
    precommit = ApprovedHierarchicalDiscoveryBatchPrecommit.create(packet)
    coordinator = object.__new__(ApprovedHierarchicalDiscoveryBatchCoordinator)
    coordinator.precommit = precommit
    batch_path = root / "batch.json"
    _write_wrapper(
        batch_path,
        schema="hierarchical_all_evidence_runner_batch_packet_v1",
        body={"approval_sha256": precommit.approval_sha256, "packet": packet},
    )
    prepared = object.__new__(PreparedHierarchicalDiscoveryBatch)
    prepared_folds = []
    for outer_fold in (1, 2):
        fold = object.__new__(PreparedHierarchicalDiscoveryFold)
        object.__setattr__(fold, "outer_fold", outer_fold)
        prepared_folds.append(fold)
    object.__setattr__(prepared, "coordinator", coordinator)
    object.__setattr__(prepared, "folds", tuple(prepared_folds))
    object.__setattr__(prepared, "input_manifest_sha256", input_sha256)
    object.__setattr__(prepared, "input_manifest_path", input_path)
    object.__setattr__(prepared, "batch_packet_path", batch_path)
    if issue_prepared:
        _issue_prepared_hierarchy_capability(prepared)
    artifacts = {
        "dataset": dataset,
        "legacy": legacy,
        "tfidf": tfidf,
        "primary": primary,
        "candidate": candidate,
        "query": query,
        "orphan": orphan,
    }
    return handoff, runner, prepared, gate_provider, artifacts


@pytest.mark.parametrize(
    "mutation",
    [
        lambda body: body["dataset"].__setitem__("sha256", "0" * 64),
        lambda body: body["legacy_handoff"].__setitem__("sha256", "1" * 64),
        lambda body: body["tfidf_handoff"].__setitem__("sha256", "2" * 64),
        lambda body: body.__setitem__("spent_evidence_provider", None),
        lambda body: body.__setitem__("hierarchical_discovery_config", {"changed": True}),
        lambda body: body.__setitem__("outer_folds", [{"outer_fold": 2}, {"outer_fold": 1}]),
        lambda body: body["outer_folds"].append("ignored-non-mapping-row"),
    ],
    ids=(
        "dataset",
        "legacy",
        "tfidf",
        "provider",
        "config",
        "fold_order",
        "malformed_extra_fold_row",
    ),
)
def test_capability_rejects_self_consistent_generic_preparation_binding_drift(
    tmp_path: Path,
    mutation,
):
    handoff, runner, prepared, _gate, _artifacts = _case(
        tmp_path,
        input_mutation=mutation,
    )
    with pytest.raises(ValueError, match="not bound to this Stage 1 provider"):
        prepare_internal_hierarchy_execution_capability(
            handoff=handoff,
            runner=runner,
            prepared_batch=prepared,
        )


def test_capability_rejects_unissued_batch_and_non_runner(tmp_path: Path):
    handoff, runner, prepared, _gate, _artifacts = _case(
        tmp_path,
        issue_prepared=False,
    )
    with pytest.raises(TypeError, match="concrete AllEvidenceFusionRunner"):
        prepare_internal_hierarchy_execution_capability(
            handoff=handoff,
            runner=SimpleNamespace(),
            prepared_batch=prepared,
        )
    with pytest.raises(RuntimeError, match="no fresh in-memory capability"):
        prepare_internal_hierarchy_execution_capability(
            handoff=handoff,
            runner=runner,
            prepared_batch=prepared,
        )


def test_prepared_capability_itself_is_one_use(tmp_path: Path):
    handoff, runner, prepared, _gate, _artifacts = _case(tmp_path)
    prepare_internal_hierarchy_execution_capability(
        handoff=handoff,
        runner=runner,
        prepared_batch=prepared,
    )
    with pytest.raises(RuntimeError, match="no fresh in-memory capability"):
        prepare_internal_hierarchy_execution_capability(
            handoff=handoff,
            runner=runner,
            prepared_batch=prepared,
        )


@pytest.mark.parametrize(
    "artifact_name",
    ("dataset", "legacy", "tfidf", "primary", "candidate", "query", "orphan"),
)
def test_authorization_rejects_every_runtime_file_mutated_after_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
):
    monkeypatch.setattr(
        handoff_module,
        "GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY",
        True,
    )
    handoff, runner, prepared, _gate, artifacts = _case(tmp_path)
    capability = prepare_internal_hierarchy_execution_capability(
        handoff=handoff,
        runner=runner,
        prepared_batch=prepared,
    )
    artifacts[artifact_name].write_bytes(b"post-preparation replacement")
    with pytest.raises(ValueError, match="changed|runtime identity"):
        internal_hierarchy_execution_authorization(
            handoff=handoff,
            prepared_capability=capability,
        )


def test_authorization_rejects_runtime_provider_object_and_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        handoff_module,
        "GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY",
        True,
    )
    handoff, runner, prepared, gate, _artifacts = _case(tmp_path)
    capability = prepare_internal_hierarchy_execution_capability(
        handoff=handoff,
        runner=runner,
        prepared_batch=prepared,
    )
    gate.revision += 1
    with pytest.raises(ValueError, match="runtime identity changed"):
        internal_hierarchy_execution_authorization(
            handoff=handoff,
            prepared_capability=capability,
        )


def _authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        handoff_module,
        "GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY",
        True,
    )
    handoff, runner, prepared, _gate, _artifacts = _case(tmp_path)
    capability = prepare_internal_hierarchy_execution_capability(
        handoff=handoff,
        runner=runner,
        prepared_batch=prepared,
    )
    authorization = internal_hierarchy_execution_authorization(
        handoff=handoff,
        prepared_capability=capability,
    )
    return handoff, runner, prepared, authorization


def test_execution_rejects_copied_prepared_batch_and_wrong_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _handoff, runner, prepared, authorization = _authorization(tmp_path, monkeypatch)
    copied = object.__new__(PreparedHierarchicalDiscoveryBatch)
    object.__setattr__(copied, "coordinator", prepared.coordinator)
    object.__setattr__(copied, "folds", prepared.folds)
    object.__setattr__(copied, "input_manifest_sha256", prepared.input_manifest_sha256)
    with pytest.raises(ValueError, match="another runner or prepared batch"):
        authorization._execute_for_prepared_batch(
            prepared_batch=copied,
            runner=runner,
        )
    wrong_runner = object.__new__(AllEvidenceFusionRunner)
    with pytest.raises(ValueError, match="another runner or prepared batch"):
        authorization._execute_for_prepared_batch(
            prepared_batch=prepared,
            runner=wrong_runner,
        )


def test_execution_rejects_coordinator_substitution_and_method_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _handoff, runner, prepared, authorization = _authorization(tmp_path, monkeypatch)
    replacement = object.__new__(ApprovedHierarchicalDiscoveryBatchCoordinator)
    replacement.precommit = prepared.coordinator.precommit
    original = prepared.coordinator
    object.__setattr__(prepared, "coordinator", replacement)
    with pytest.raises(ValueError, match="coordinator changed"):
        authorization._execute_for_prepared_batch(
            prepared_batch=prepared,
            runner=runner,
        )
    object.__setattr__(prepared, "coordinator", original)
    original.execute = lambda **_kwargs: None
    with pytest.raises(ValueError, match="instance method override"):
        authorization._execute_for_prepared_batch(
            prepared_batch=prepared,
            runner=runner,
        )


def test_execution_rechecks_file_and_exact_provider_object_after_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _handoff, runner, prepared, authorization = _authorization(tmp_path, monkeypatch)
    runner.dataset_path.write_bytes(b"replacement after authorization")
    with pytest.raises(ValueError, match="runtime identity changed"):
        authorization._execute_for_prepared_batch(
            prepared_batch=prepared,
            runner=runner,
        )

    # A distinct provider object with a byte-identical identity is still not
    # the process-local object that preparation authorized.
    runner.dataset_path.write_text(json.dumps({"artifact": "dataset"}) + "\n")
    replacement_gate = _IdentityObject("shared-gate-provider")
    runner.review_gate_source_provider = replacement_gate
    runner.review_gate_feature_bank_provider = replacement_gate
    with pytest.raises(ValueError, match="runtime object changed"):
        authorization._execute_for_prepared_batch(
            prepared_batch=prepared,
            runner=runner,
        )


def test_execution_rejects_byte_identical_precommit_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _handoff, runner, prepared, authorization = _authorization(tmp_path, monkeypatch)
    prepared.coordinator.precommit = ApprovedHierarchicalDiscoveryBatchPrecommit.create(
        prepared.coordinator.precommit.packet
    )
    with pytest.raises(ValueError, match="coordinator changed"):
        authorization._execute_for_prepared_batch(
            prepared_batch=prepared,
            runner=runner,
        )


def test_execution_rejects_prepared_fold_tuple_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _handoff, runner, prepared, authorization = _authorization(tmp_path, monkeypatch)
    object.__setattr__(prepared, "folds", tuple([*prepared.folds]))
    with pytest.raises(ValueError, match="fold objects changed"):
        authorization._execute_for_prepared_batch(
            prepared_batch=prepared,
            runner=runner,
        )


def test_execution_rejects_noncanonical_result_and_consumes_attempt_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _handoff, runner, prepared, authorization = _authorization(tmp_path, monkeypatch)
    capability = authorization._prepared_capability

    def fake_execute(_coordinator, *, approved_batch_sha256):
        assert approved_batch_sha256 == prepared.approval_sha256
        return SimpleNamespace(validate_authentication=lambda: None)

    # Exercise the final result-type guard independently of the coordinator
    # implementation guard. All three references must agree before invocation.
    monkeypatch.setattr(ApprovedHierarchicalDiscoveryBatchCoordinator, "execute", fake_execute)
    monkeypatch.setattr(handoff_module, "_CANONICAL_COORDINATOR_EXECUTE", fake_execute)
    capability._coordinator_execute_function = fake_execute
    with pytest.raises(TypeError, match="noncanonical batch result"):
        authorization._execute_for_prepared_batch(
            prepared_batch=prepared,
            runner=runner,
        )
    with pytest.raises(RuntimeError, match="already consumed"):
        authorization._execute_for_prepared_batch(
            prepared_batch=prepared,
            runner=runner,
        )


def test_typed_authorization_low_level_consumption_is_one_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _handoff, _runner, _prepared, authorization = _authorization(tmp_path, monkeypatch)
    assert authorization.consume_for_execution() is authorization
    with pytest.raises(RuntimeError, match="already consumed"):
        authorization.consume_for_execution()


def test_authorization_has_no_path_replay_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _handoff, _runner, _prepared, authorization = _authorization(tmp_path, monkeypatch)
    payload = authorization.as_dict()
    assert payload["caller_replay_registrations_accepted"] is False
    assert payload["execution_sources"] == "exact_runner_held_authenticated_providers"
    assert not any(
        "replay" in key for key in payload if key != "caller_replay_registrations_accepted"
    )
    assert payload["exact_coordinator_object_bound"] is True
    assert payload["canonical_unbound_coordinator_execute_required"] is True
    assert payload["exact_batch_result_type_required"] is True


def test_production_one_shot_rejects_caller_digest_and_has_no_replay_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        handoff_module,
        "GENUINE_HIERARCHY_NATIVE_PROOF_VALIDATION_READY",
        True,
    )
    handoff, runner, _prepared, _gate, _artifacts = _case(tmp_path)
    runner.hierarchical_discovery_approved_batch_sha256 = "0" * 64
    with pytest.raises(ValueError, match="caller-supplied digest"):
        run_internal_production_stage1_hierarchy_one_shot(
            handoff=handoff,
            runner=runner,
        )
    assert (
        "authoritative_execution_replay_arguments"
        not in inspect.signature(run_internal_production_stage1_hierarchy_one_shot).parameters
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_internal_production_stage1_hierarchy_one_shot(
            handoff=handoff,
            runner=runner,
            authoritative_execution_replay_arguments=[],
        )


def test_prepared_execution_rejects_serialized_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _handoff, runner, prepared, authorization = _authorization(tmp_path, monkeypatch)
    with pytest.raises(TypeError, match="exact typed authorization"):
        prepared.execute_with_internal_authorization(
            authorization=authorization.as_dict(),
            runner=runner,
        )
