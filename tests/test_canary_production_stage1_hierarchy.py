from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import scripts.canary_production_stage1_hierarchy as canary
from oci.inference.all_evidence_discovery_interfaces import (
    BOW_NUISANCE,
    BOW_R_LOSS,
    DiscoveryEvidenceItem,
    canonical_json,
    content_sha256,
    render_interpret_evidence_chunk_messages,
)
from oci.inference.approved_hierarchical_discovery_agent import (
    _PerCallMetadataAuthenticatingRunner,
)
from oci.inference.hierarchical_all_architecture_discovery import (
    HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING,
    INTERPRET_CHUNK_JOB,
    DiscoveryJobSettings,
    DiscoveryJsonJob,
    HierarchicalAllArchitectureDiscoveryOrchestrator,
    hierarchical_discovery_implementation_bundle,
)
from oci.inference.production_stage1_hierarchy_one_shot import (
    ProductionStage1HierarchyOneShotOptions,
)

CAMUS_ENDPOINT = "http://camus:8010/v1"
CAMUS_MODEL = "RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
LOCAL_ENDPOINT = "http://localhost:2345/v1"
LOCAL_MODEL = "local/test-model"


def _evidence(
    *,
    evidence_id: str,
    source_family: str = BOW_NUISANCE,
    clue: str = "age",
) -> DiscoveryEvidenceItem:
    suffix = evidence_id.rsplit(".", 1)[-1]
    return DiscoveryEvidenceItem(
        evidence_id=evidence_id,
        source_family=source_family,
        observable_axes=("heterogeneity",),
        content={"readable_clue": clue},
        member_ids=(f"member.canary.{suffix}",),
    )


def _interpret_job(
    *,
    evidence: DiscoveryEvidenceItem,
    chunk_id: str,
    explanation: str = "Interpret the exact architecture-local readable clue.",
) -> DiscoveryJsonJob:
    return DiscoveryJsonJob.create(
        job_kind=INTERPRET_CHUNK_JOB,
        scope=f"{evidence.source_family}.chunk_000",
        dependencies=(),
        settings=DiscoveryJobSettings.selector(),
        messages=render_interpret_evidence_chunk_messages(
            family_explanation=explanation,
            evidence=(evidence,),
        ),
        input_bindings={
            "catalog_sha256": "a" * 64,
            "chunk_plan_sha256": "b" * 64,
            "chunk_id": chunk_id,
            "source_family": evidence.source_family,
            HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING: (
                hierarchical_discovery_implementation_bundle()["implementation_bundle_sha256"]
            ),
        },
    )


class _Atom:
    def __init__(self, item: DiscoveryEvidenceItem):
        self.evidence_id = item.evidence_id
        self._item = item

    def as_discovery_item(self) -> DiscoveryEvidenceItem:
        return self._item


class _Cache:
    def __init__(self) -> None:
        self.begin_calls: list[dict[str, Any]] = []
        self.replay_calls: list[dict[str, Any]] = []
        self.store_calls: list[dict[str, Any]] = []

    @property
    def execution_metadata(self) -> tuple[()]:
        return ()

    def identity(self) -> dict[str, Any]:
        body = {
            "schema_version": "offline_canary_cache_identity_v1",
            "mode": "read_write_immutable",
            "validated_only": True,
        }
        return {**body, "identity_sha256": content_sha256(body)}

    def begin_execution(self, **kwargs: Any) -> None:
        self.begin_calls.append(copy.deepcopy(kwargs))

    def replay_validated(self, **kwargs: Any) -> None:
        self.replay_calls.append(copy.deepcopy(kwargs))
        return None

    def store_validated(self, **kwargs: Any) -> None:
        self.store_calls.append(copy.deepcopy(kwargs))


class _Orchestrator:
    def __init__(self, *, job: DiscoveryJsonJob, runner_identity: Mapping[str, Any]):
        self.initial_job_ledger = SimpleNamespace(jobs=(job,))
        self.job_cache = _Cache()
        self.precommit = SimpleNamespace(precommit_sha256="c" * 64)
        self.runner_identity = copy.deepcopy(dict(runner_identity))
        self.implementation_bundle_sha256 = hierarchical_discovery_implementation_bundle()[
            "implementation_bundle_sha256"
        ]
        self.config = SimpleNamespace(max_rendered_prompt_bytes=10_000_000)
        self.implementation_checks: list[dict[str, Any]] = []

    def _assert_runner_identity(self, runner: Any) -> None:
        assert canonical_json(runner.identity()) == canonical_json(self.runner_identity)

    def _assert_implementation_bundle_unchanged(self, **kwargs: Any) -> None:
        self.implementation_checks.append(copy.deepcopy(kwargs))

    def _run(self, **kwargs: Any):
        assert isinstance(kwargs["runner"], _PerCallMetadataAuthenticatingRunner)
        return HierarchicalAllArchitectureDiscoveryOrchestrator._run(self, **kwargs)


class _Agent:
    def __init__(self, *, runner: Any, orchestrator: _Orchestrator):
        self.runner = runner
        self._orchestrator = orchestrator

    def _assert_unchanged(self):
        return self.runner.identity(), self._orchestrator


def _fold(
    *,
    outer_fold: int,
    runner: Any,
    evidence: DiscoveryEvidenceItem,
    job: DiscoveryJsonJob,
    orchestrator: _Orchestrator,
) -> Any:
    chunk_id = str(job.input_bindings["chunk_id"])
    chunk = SimpleNamespace(
        chunk_id=chunk_id,
        source_family=evidence.source_family,
        evidence=[{"evidence_id": evidence.evidence_id}],
    )
    return SimpleNamespace(
        outer_fold=outer_fold,
        agent=_Agent(runner=runner, orchestrator=orchestrator),
        catalog=SimpleNamespace(atoms=(_Atom(evidence),)),
        chunk_plan=SimpleNamespace(chunks=(chunk,)),
    )


def _runner_identity(
    *,
    endpoint: str = CAMUS_ENDPOINT,
    model_name: str = CAMUS_MODEL,
) -> dict[str, Any]:
    body = {
        "schema_version": "offline_canary_runner_v1",
        "endpoint_urls": [endpoint],
        "model": {
            "name": model_name,
            "resolution": "explicit_only_no_autodiscovery",
        },
        "retry": {"max_retries": 0, "max_attempts": 1},
        "max_tokens": canary.EXACT_MAX_TOKENS,
        "response_semantics": {
            "selector_thinking": {
                "enabled": True,
                "thinking_token_budget": 5_000,
            },
            "extraction_thinking": {
                "enabled": False,
                "thinking_token_budget_field": "omitted",
            },
        },
    }
    return {**body, "identity_sha256": content_sha256(body)}


class _MetadataRunner:
    def __init__(
        self,
        *,
        identity: Mapping[str, Any],
        evidence: DiscoveryEvidenceItem,
        endpoint: str,
        model_name: str,
        response_model: str | None = None,
        second_response_model: str | None = None,
        finish_reason: str | None = "stop",
        invalid_first_wire: bool = False,
    ):
        self._identity = copy.deepcopy(dict(identity))
        self._evidence = evidence
        self._endpoint = endpoint
        self._model_name = model_name
        self._response_model = model_name if response_model is None else response_model
        self._second_response_model = second_response_model
        self._finish_reason = finish_reason
        self._invalid_first_wire = invalid_first_wire
        self._metadata: list[dict[str, Any]] = []
        self.calls: list[DiscoveryJsonJob] = []
        self.closed = False

    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._identity)

    @property
    def execution_metadata(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._metadata))

    def run_json(self, *, job: DiscoveryJsonJob) -> Mapping[str, Any]:
        self.calls.append(job)
        member_id = self._evidence.member_ids[0]
        if self._invalid_first_wire and len(self.calls) == 1:
            response = {"evidence_dispositions": {}}
        else:
            response = {
                "evidence_dispositions": {
                    self._evidence.evidence_id: {
                        "evidence_findings": [],
                        "member_dispositions": {member_id: {"findings": []}},
                        "reason": "No specific patient concept is supported.",
                    }
                }
            }
        request_sha256 = content_sha256(job.as_dict())
        response_sha256 = content_sha256(response)
        raw = canonical_json(response).encode("utf-8")
        attempt = {
            "attempt_number": 1,
            "endpoint": self._endpoint,
            "model": self._model_name,
            "request_sha256": request_sha256,
            "runner_identity_sha256": self._identity["identity_sha256"],
            "outcome": "success",
            "retryable": False,
            "will_retry": False,
            "response_id": "offline-fake-response",
            "response_model": (
                self._second_response_model
                if len(self.calls) == 2 and self._second_response_model is not None
                else self._response_model
            ),
            "usage": {},
            "content_sha256": hashlib.sha256(raw).hexdigest(),
            "reasoning_hashes": {},
            "raw_transport_bytes": len(raw),
            "parsed_response_sha256": response_sha256,
        }
        if self._finish_reason is not None:
            attempt["finish_reason"] = self._finish_reason
        self._metadata.append(
            {
                "job_id": job.job_id,
                "job_kind": job.job_kind,
                "request_sha256": request_sha256,
                "runner_identity_sha256": self._identity["identity_sha256"],
                "outcome": "success",
                "parsed_response_sha256": response_sha256,
                "attempts": [attempt],
            }
        )
        return response

    def close(self) -> None:
        self.closed = True


class _ProductionRunner:
    def __init__(self, *, hierarchy_runner: _MetadataRunner, prepared: Any):
        self.hierarchical_discovery_runner = hierarchy_runner
        self.hierarchical_discovery_approved_batch_sha256 = None
        self.config = SimpleNamespace(
            fusion_enable_thinking=True,
            fusion_thinking_token_budget=5_000,
            fusion_max_tokens=25_000,
            extraction_enable_thinking=False,
        )
        self._prepared = prepared
        self.preparation_calls = 0

    def prepare_hierarchical_discovery_batch(self):
        self.preparation_calls += 1
        return self._prepared


class _Handoff:
    def __init__(self) -> None:
        self.inputs = SimpleNamespace(bundle_sha256="d" * 64)
        body = {
            "manual_digest_approval_required": False,
            "raw_all_architecture_prompt_allowed": False,
            "per_architecture_interpretation_required": True,
            "all_ten_architectures_required": True,
        }
        self._value = {**body, "content_sha256": content_sha256(body)}

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._value)


def _options(
    tmp_path: Path,
    *,
    endpoint: str = CAMUS_ENDPOINT,
    model_name: str = CAMUS_MODEL,
) -> ProductionStage1HierarchyOneShotOptions:
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    bundle = bundle_root / "bundle_manifest.json"
    bundle.write_text("{}\n", encoding="utf-8")
    return ProductionStage1HierarchyOneShotOptions(
        bundle_manifest_path=bundle,
        output_dir=tmp_path / "canary_output",
        preparation_dir=tmp_path / "canary_preparation",
        attestation_dir=tmp_path / "canary_attestation",
        endpoint=endpoint,
        model_name=model_name,
        review_rounds=1,
        proposal_max_tokens=25_000,
        extraction_max_tokens=25_000,
        proposal_schema_repair_attempts=1,
        request_max_retries=0,
    )


def _install_fake_production_graph(
    monkeypatch: pytest.MonkeyPatch,
    *,
    options: ProductionStage1HierarchyOneShotOptions,
    response_model: str | None = None,
    second_response_model: str | None = None,
    finish_reason: str | None = "stop",
    invalid_first_wire: bool = False,
) -> tuple[_MetadataRunner, _Orchestrator, _ProductionRunner]:
    identity = _runner_identity(
        endpoint=options.endpoint,
        model_name=options.model_name,
    )
    evidence = _evidence(evidence_id="evidence.canary.success")
    job = _interpret_job(evidence=evidence, chunk_id="chunk.canary.success")
    hierarchy_runner = _MetadataRunner(
        identity=identity,
        evidence=evidence,
        endpoint=options.endpoint,
        model_name=options.model_name,
        response_model=response_model,
        second_response_model=second_response_model,
        finish_reason=finish_reason,
        invalid_first_wire=invalid_first_wire,
    )
    orchestrator = _Orchestrator(job=job, runner_identity=identity)
    prepared = SimpleNamespace(
        folds=(
            _fold(
                outer_fold=1,
                runner=hierarchy_runner,
                evidence=evidence,
                job=job,
                orchestrator=orchestrator,
            ),
        )
    )
    production_runner = _ProductionRunner(
        hierarchy_runner=hierarchy_runner,
        prepared=prepared,
    )
    handoff = _Handoff()

    def load_handoff(path: Path, **kwargs: Any):
        assert path == options.bundle_manifest_path
        assert kwargs == {
            "review_rounds": 1,
            "interaction_inner_folds": 3,
            "tfidf_nested_calibration_folds": 3,
        }
        return handoff

    def build_runner(**kwargs: Any):
        assert kwargs["handoff"] is handoff
        assert kwargs["options"] is options
        assert kwargs["endpoint"] == options.endpoint
        assert "model_identity" not in kwargs
        return production_runner

    monkeypatch.setattr(canary, "load_production_stage1_hierarchy_handoff", load_handoff)
    monkeypatch.setattr(canary, "build_production_stage1_hierarchy_runner", build_runner)
    return hierarchy_runner, orchestrator, production_runner


def _walk_mapping(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mapping(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mapping(child)


def test_cli_exposes_no_override_for_fixed_safety_contract(tmp_path: Path) -> None:
    parser = canary.build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--model-identity-json" not in option_strings
    forbidden_fragments = (
        "approval",
        "digest",
        "replay",
        "oracle",
        "prediction",
        "max-tokens",
        "retry",
        "repair",
    )
    assert not {
        option
        for option in option_strings
        if any(fragment in option for fragment in forbidden_fragments)
    }

    options = _options(tmp_path)
    argv = [
        "--bundle-manifest",
        str(options.bundle_manifest_path),
        "--scratch-output-dir",
        str(options.output_dir),
        "--hierarchical-preparation-dir",
        str(options.preparation_dir),
        "--report-dir",
        str(options.attestation_dir),
        "--endpoint",
        CAMUS_ENDPOINT,
        "--model",
        CAMUS_MODEL,
        "--review-rounds",
        "1",
    ]
    parsed = canary.options_from_args(parser.parse_args(argv))
    assert parsed.proposal_max_tokens == 25_000
    assert parsed.extraction_max_tokens == 25_000
    assert parsed.request_max_retries == 0
    assert parsed.proposal_schema_repair_attempts == 1

    local_argv = list(argv)
    local_argv[local_argv.index(CAMUS_ENDPOINT)] = LOCAL_ENDPOINT
    local_argv[local_argv.index(CAMUS_MODEL)] = LOCAL_MODEL
    local = canary.options_from_args(parser.parse_args(local_argv))
    assert local.endpoint == LOCAL_ENDPOINT
    assert local.model_name == LOCAL_MODEL

    bad_endpoint = list(argv)
    bad_endpoint[bad_endpoint.index(CAMUS_ENDPOINT)] = f"{CAMUS_ENDPOINT},{LOCAL_ENDPOINT}"
    with pytest.raises(ValueError, match="single|pool|comma"):
        canary.options_from_args(parser.parse_args(bad_endpoint))


def test_selection_uses_smallest_real_architecture_pure_prompt() -> None:
    identity = _runner_identity()
    runner = SimpleNamespace(identity=lambda: copy.deepcopy(identity))

    long_evidence = _evidence(
        evidence_id="evidence.canary.long",
        clue="a much longer architecture-local readable clue " * 8,
    )
    short_evidence = _evidence(evidence_id="evidence.canary.short", clue="age")
    long_job = _interpret_job(evidence=long_evidence, chunk_id="chunk.canary.long")
    short_job = _interpret_job(evidence=short_evidence, chunk_id="chunk.canary.short")
    long_orchestrator = _Orchestrator(job=long_job, runner_identity=identity)
    short_orchestrator = _Orchestrator(job=short_job, runner_identity=identity)
    prepared = SimpleNamespace(
        folds=(
            _fold(
                outer_fold=1,
                runner=runner,
                evidence=long_evidence,
                job=long_job,
                orchestrator=long_orchestrator,
            ),
            _fold(
                outer_fold=2,
                runner=runner,
                evidence=short_evidence,
                job=short_job,
                orchestrator=short_orchestrator,
            ),
        )
    )

    selected = canary._select_smallest_initial_interpretation_job(
        prepared_batch=prepared,
        production_hierarchy_runner=runner,
    )

    assert selected.job.job_id == short_job.job_id
    assert selected.source_family == BOW_NUISANCE
    assert selected.evidence == (short_evidence,)
    assert selected.rendered_message_bytes < len(long_job.rendered_messages_bytes)


def test_selection_rejects_architecture_mismatch() -> None:
    identity = _runner_identity()
    runner = SimpleNamespace(identity=lambda: copy.deepcopy(identity))
    evidence = _evidence(evidence_id="evidence.canary.mismatch")
    job = _interpret_job(evidence=evidence, chunk_id="chunk.canary.mismatch")
    orchestrator = _Orchestrator(job=job, runner_identity=identity)
    fold = _fold(
        outer_fold=1,
        runner=runner,
        evidence=evidence,
        job=job,
        orchestrator=orchestrator,
    )
    fold.chunk_plan.chunks[0].source_family = BOW_R_LOSS

    with pytest.raises(ValueError, match="different architectures"):
        canary._select_smallest_initial_interpretation_job(
            prepared_batch=SimpleNamespace(folds=(fold,)),
            production_hierarchy_runner=runner,
        )


def test_run_canary_uses_one_authenticated_production_job_and_emits_hashes_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    runner, orchestrator, production_runner = _install_fake_production_graph(
        monkeypatch,
        options=options,
    )

    summary = canary.run_canary(options)

    assert summary["status"] == "accepted"
    assert summary["remote_response_count"] == 1
    assert len(runner.calls) == 1
    assert production_runner.preparation_calls == 1
    assert runner.closed is True
    assert len(orchestrator.job_cache.begin_calls) == 1
    assert len(orchestrator.job_cache.replay_calls) == 1
    assert len(orchestrator.job_cache.store_calls) == 1
    assert not (options.output_dir / "frozen_predictions.parquet").exists()
    assert not (options.output_dir / "immutable_run_manifest.json").exists()

    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    assert report["schema_version"] == canary.CANARY_REPORT_SCHEMA
    body = report["body"]
    assert body["endpoint"] == CAMUS_ENDPOINT
    assert body["model"] == CAMUS_MODEL
    assert "served_deployment_identity" not in body
    assert body["authorization_role"] == "non_authorizing_operational_runtime_check"
    assert body["settings"] == {
        "extraction_thinking_enabled": False,
        "max_tokens": 25_000,
        "maximum_schema_repairs": 1,
        "selector_thinking_enabled": True,
        "selector_thinking_token_budget": 5_000,
        "transport_retries": 0,
    }
    assert body["remote_response_count"] == 1
    assert body["transport_metadata"][0]["attempts"][0]["response_model"] == (CAMUS_MODEL)
    assert body["transport_metadata"][0]["attempts"][0]["finish_reason"] == "stop"
    assert body["prediction_path_constructed"] is False
    assert body["oracle_path_constructed"] is False
    assert body["validation"]["job_cache_identity_sha256"] == (
        orchestrator.job_cache.identity()["identity_sha256"]
    )
    forbidden = canary._FORBIDDEN_OUTPUT_KEYS
    assert all(not (set(row) & forbidden) for row in _walk_mapping(report))
    serialized = canonical_json(report)
    assert "No specific patient concept is supported" not in serialized
    assert "readable_clue" not in serialized


def test_run_canary_binds_an_intentional_local_endpoint_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(
        tmp_path,
        endpoint=LOCAL_ENDPOINT,
        model_name=LOCAL_MODEL,
    )
    runner, orchestrator, _production_runner = _install_fake_production_graph(
        monkeypatch,
        options=options,
    )

    summary = canary.run_canary(options)

    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    body = report["body"]
    assert body["endpoint"] == LOCAL_ENDPOINT
    assert body["model"] == LOCAL_MODEL
    attempt = body["transport_metadata"][0]["attempts"][0]
    assert attempt["endpoint"] == LOCAL_ENDPOINT
    assert attempt["model"] == LOCAL_MODEL
    assert attempt["response_model"] == LOCAL_MODEL
    assert attempt["finish_reason"] == "stop"
    assert len(runner.calls) == 1
    assert len(orchestrator.job_cache.store_calls) == 1


def test_run_canary_allows_only_the_single_authenticated_schema_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    runner, orchestrator, _production_runner = _install_fake_production_graph(
        monkeypatch,
        options=options,
        invalid_first_wire=True,
    )

    summary = canary.run_canary(options)

    assert summary["remote_response_count"] == 2
    assert len(runner.calls) == 2
    assert len(orchestrator.job_cache.store_calls) == 1
    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    body = report["body"]
    assert body["validation"]["response_attempt_outcomes"] == [
        "local_json_schema_validation_failure",
        "validated_response",
    ]
    assert all(len(record["attempts"]) == 1 for record in body["transport_metadata"])
    assert all(
        record["attempts"][0]["response_model"] == CAMUS_MODEL
        and record["attempts"][0]["finish_reason"] == "stop"
        for record in body["transport_metadata"]
    )


def test_run_canary_authenticates_the_repair_response_before_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    runner, orchestrator, _production_runner = _install_fake_production_graph(
        monkeypatch,
        options=options,
        invalid_first_wire=True,
        second_response_model="substituted-repair-model",
    )

    with pytest.raises(ValueError, match="response model differs"):
        canary.run_canary(options)

    assert len(runner.calls) == 2
    assert runner.closed is True
    assert orchestrator.job_cache.store_calls == []
    assert not options.attestation_dir.exists()


@pytest.mark.parametrize(
    ("response_model", "finish_reason", "error"),
    [
        ("substituted-model", "stop", "response model differs"),
        (None, "length", "finish_reason must be exactly 'stop'"),
        (None, None, "finish_reason must be exactly 'stop'"),
    ],
)
def test_run_canary_rejects_unauthenticated_response_metadata_before_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_model: str | None,
    finish_reason: str | None,
    error: str,
) -> None:
    options = _options(tmp_path)
    runner, orchestrator, _production_runner = _install_fake_production_graph(
        monkeypatch,
        options=options,
        response_model=response_model,
        finish_reason=finish_reason,
    )

    with pytest.raises(ValueError, match=error):
        canary.run_canary(options)

    assert len(runner.calls) == 1
    assert runner.closed is True
    assert orchestrator.job_cache.store_calls == []
    assert not options.attestation_dir.exists()
    assert not (options.output_dir / "frozen_predictions.parquet").exists()
