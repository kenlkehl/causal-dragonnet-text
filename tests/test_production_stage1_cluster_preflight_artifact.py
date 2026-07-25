import copy
import json
import stat
from pathlib import Path

import pytest

import oci.inference.production_stage1_cluster_preflight_artifact as artifact_module
from oci.inference.production_stage1_cluster_preflight_artifact import (
    CLUSTER_PREFLIGHT_AUDIT_NAME,
    CLUSTER_PREFLIGHT_MANIFEST_NAME,
    CLUSTER_PREFLIGHT_REQUEST_NAME,
    load_production_stage1_cluster_preflight_artifact,
    seal_production_stage1_cluster_preflight_artifact,
)


def _addressed(body):
    return {**body, "content_sha256": artifact_module._sha256_json(body)}


def _rehash_addressed(value):
    body = {
        key: child
        for key, child in value.items()
        if key != "content_sha256"
    }
    value["content_sha256"] = artifact_module._sha256_json(body)


def _rehash_request(value):
    body = {
        key: child
        for key, child in value.items()
        if key != "request_sha256"
    }
    value["request_sha256"] = artifact_module._sha256_json(body)


def _fixture_values(scope_count=40):
    scopes = []
    for index in range(scope_count):
        scope_id = f"scope_{index:03d}"
        scopes.append(
            {
                "scope_id": scope_id,
                "scope_kind": (
                    "full_outer"
                    if index < 5
                    else "exact_inner" if index < 30 else "cumulative_spent"
                ),
                "outer_fold": index % 5,
                "inner_fold": None if index < 5 else index % 5,
                "context_epoch": None if index < 30 else 1 + index % 2,
                "provider_inner_fold": None if index < 30 else index % 5,
                "fit_row_count": 800 if index < 5 else 640,
                "fit_row_order_fingerprint": artifact_module._sha256_json(
                    ["fit", index]
                ),
                "heldout_row_count": 200 if index < 5 else 160,
                "heldout_row_order_fingerprint": artifact_module._sha256_json(
                    ["heldout", index]
                ),
                "cluster_fit_identity": {
                    "scope_id": scope_id,
                    "components": artifact_module._sha256_json(["components", index]),
                },
            }
        )
    audit_body = {
        "schema_version": "fixture_cluster_audit",
        "scope_count": scope_count,
        "scope_order": [row["scope_id"] for row in scopes],
        "scopes": scopes,
    }
    audit = _addressed(audit_body)
    scientific_scope_plan = {
        "schema_version": "fixture_scientific_scope_plan",
        "scope_order": audit["scope_order"],
    }
    scope_plan_body = {
        "schema_version": "fixture_scope_plan",
        "scientific_content_sha256": artifact_module._sha256_json(
            scientific_scope_plan
        ),
        "scope_order": audit["scope_order"],
        "gpu_ids": [0, 1],
        "scope_workers_per_gpu": 1,
        "assignments": list(reversed(audit["scope_order"])),
    }
    cache_build_identity = {
        "cache_path": "/fixture/cache",
        "local_model_tree_sha256": "8" * 64,
        "cache_files": {"embeddings.npy": {"sha256": "9" * 64}},
    }
    request_body = {
        "schema_version": "fixture_stage1_request",
        "dataset": {"path": "/fixture/data", "sha256": "1" * 64},
        "source_config": {"path": "/fixture/profile", "sha256": "2" * 64},
        "effective_stage1_config": {
            "seed": 42,
            "dataset_path": "/fixture/data",
            "architecture": {
                "agentic_feature_search": {
                    "agent_server_url": "http://fixture.invalid/v1",
                    "agent_model_name": "fixture-model",
                },
                "htr_sentence_model": "/fixture/htr",
                "multi_model_forest": {
                    "embedding_contrast": {
                        "cache_dir": "/fixture/cache",
                        "external_corpus_cache_dirs": [],
                        "model_name": "fixture-content-addressed-encoder",
                    }
                },
            },
        },
        "embedding_cache": {
            "path": "/fixture/cache",
            "identity": {"sha256": "3" * 64},
            "production_cache_build_identity": cache_build_identity,
            "authenticated_relocation": {
                "schema_version": "fixture_cache_relocation",
                "relocator_version": "fixture_relocator",
                "relocator_code_sha256": "a" * 64,
                "authenticated_tree_code_sha256": "b" * 64,
                "root": "/fixture/relocation",
                "cache_dir": "/fixture/relocation/cache",
                "prepared_cohort_path": "/fixture/relocation/cohort.parquet",
                "attestation_path": "/fixture/relocation/attestation.json",
                "terminal_manifest_path": "/fixture/relocation/manifest.json",
                "row_count": 40,
                "prepared_projection_sha256": "c" * 64,
                "source_cache_identity_sha256": "d" * 64,
                "cache_build_identity": copy.deepcopy(cache_build_identity),
                "attestation_sha256": "e" * 64,
                "terminal_manifest_sha256": "f" * 64,
            },
        },
        "htr_model": {"path": "/fixture/htr", "tree_sha256": "4" * 64},
        "htr_input_nontruncation_audit": {"all_tokens_accounted_for": True},
        "split_registry_content_sha256": "5" * 64,
        "behavior_identity": _addressed({"source_tree_sha256": "6" * 64}),
        "hierarchical_discovery_contract_identity": _addressed(
            {"contract_sha256": "7" * 64}
        ),
        "architecture_contract": {"all_ten": True},
        "stage1_scope_plan": _addressed(scope_plan_body),
        "exact_inner_contract": {"complete": True},
        "query_config": {"epochs": 120},
        "semantic_witness_scientific_config": {"mode": "lossless"},
        "runtime": {"preflight_workers": 8},
        "hierarchy_spent_evidence_contract": {"review_rounds": 2},
        "security": {"oracle_accessed": False},
        "embedding_cluster_feasibility_audit": audit,
    }
    request = {
        **request_body,
        "request_sha256": artifact_module._sha256_json(request_body),
    }
    return audit, request


@pytest.fixture
def identity_validators(monkeypatch):
    monkeypatch.setattr(
        artifact_module,
        "_validate_scientific_audit",
        lambda audit, **_kwargs: copy.deepcopy(dict(audit)),
    )
    monkeypatch.setattr(
        artifact_module,
        "_validate_stage1_request",
        lambda request, **_kwargs: copy.deepcopy(dict(request)),
    )


def _seal(tmp_path, audit, request):
    return seal_production_stage1_cluster_preflight_artifact(
        output_dir=tmp_path / "preflight",
        audit=audit,
        stage1_request=request,
        config={"fixture": True},
        registry={"fixture": True},
        registry_content_sha256="5" * 64,
        embedding_cache_identity={"fixture": True},
    )


def _load(artifact, request=None):
    return load_production_stage1_cluster_preflight_artifact(
        manifest_path=artifact.manifest_path,
        config={"fixture": True},
        registry={"fixture": True},
        registry_content_sha256="5" * 64,
        embedding_cache_identity={"fixture": True},
        expected_stage1_request=request,
    )


def _make_root_writable(root: Path):
    root.chmod(stat.S_IRWXU)
    for path in root.iterdir():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_preflight_artifact_seals_all_40_ordered_scope_fits(
    tmp_path,
    identity_validators,
):
    audit, request = _fixture_values()
    artifact = _seal(tmp_path, audit, request)
    try:
        identity = artifact.identity()
        assert identity["scope_count"] == 40
        assert identity["scope_order"] == audit["scope_order"]
        assert set(path.name for path in artifact.root.iterdir()) == {
            CLUSTER_PREFLIGHT_AUDIT_NAME,
            CLUSTER_PREFLIGHT_REQUEST_NAME,
            CLUSTER_PREFLIGHT_MANIFEST_NAME,
        }
        reopened = _load(artifact, request)
        assert reopened.audit == audit
        assert reopened.stage1_request == request
    finally:
        _make_root_writable(artifact.root)


@pytest.mark.parametrize("mutation", ["missing", "extra", "audit", "reordered"])
def test_preflight_artifact_rejects_incomplete_tampered_or_reordered_bytes(
    tmp_path,
    identity_validators,
    mutation,
):
    audit, request = _fixture_values()
    artifact = _seal(tmp_path, audit, request)
    _make_root_writable(artifact.root)
    if mutation == "missing":
        artifact.audit_path.unlink()
    elif mutation == "extra":
        (artifact.root / "extra.json").write_text("{}", encoding="utf-8")
    elif mutation == "audit":
        artifact.audit_path.write_text('{"changed":true}', encoding="utf-8")
    else:
        manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        manifest["scope_records"].reverse()
        body = {
            key: value
            for key, value in manifest.items()
            if key != "content_sha256"
        }
        manifest["content_sha256"] = artifact_module._sha256_json(body)
        artifact.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    for path in artifact.root.iterdir():
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    artifact.root.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        _load(artifact, request)
    _make_root_writable(artifact.root)


def test_preflight_artifact_accepts_path_and_execution_relocation(
    tmp_path,
    identity_validators,
):
    audit, request = _fixture_values()
    substituted = copy.deepcopy(request)
    substituted["runtime"]["preflight_workers"] = 7
    substituted["dataset"]["path"] = "/different/data"
    substituted["effective_stage1_config"]["dataset_path"] = "/different/data"
    substituted["effective_stage1_config"]["architecture"][
        "htr_sentence_model"
    ] = "/different/htr"
    substituted["effective_stage1_config"]["architecture"][
        "agentic_feature_search"
    ]["agent_server_url"] = "http://different.invalid/v1"
    substituted["effective_stage1_config"]["architecture"][
        "multi_model_forest"
    ]["embedding_contrast"]["cache_dir"] = "/different/cache"
    substituted["source_config"]["path"] = "/different/profile"
    substituted["embedding_cache"]["path"] = "/different/cache"
    substituted["embedding_cache"]["production_cache_build_identity"][
        "cache_path"
    ] = "/different/cache"
    relocation = substituted["embedding_cache"]["authenticated_relocation"]
    relocation["root"] = "/different/relocation"
    relocation["cache_dir"] = "/different/relocation/cache"
    relocation["prepared_cohort_path"] = "/different/relocation/cohort.parquet"
    relocation["attestation_path"] = "/different/relocation/attestation.json"
    relocation["terminal_manifest_path"] = "/different/relocation/manifest.json"
    relocation["source_cache_identity_sha256"] = "0" * 64
    relocation["attestation_sha256"] = "1" * 64
    relocation["terminal_manifest_sha256"] = "2" * 64
    relocation["cache_build_identity"]["cache_path"] = "/different/cache"
    substituted["htr_model"]["path"] = "/different/htr"
    substituted["stage1_scope_plan"]["gpu_ids"] = [7]
    substituted["stage1_scope_plan"]["scope_workers_per_gpu"] = 3
    substituted["stage1_scope_plan"]["assignments"].reverse()
    _rehash_addressed(substituted["stage1_scope_plan"])
    _rehash_request(substituted)
    artifact = _seal(tmp_path, audit, substituted)
    try:
        reopened = _load(artifact, request)
        assert reopened.stage1_request == substituted
    finally:
        _make_root_writable(artifact.root)


def test_effective_config_projection_excludes_only_authenticated_locators():
    _audit, request = _fixture_values()
    baseline = (
        artifact_module.stage1_effective_config_scientific_compatibility_projection(
            request["effective_stage1_config"]
        )
    )
    relocated = copy.deepcopy(request["effective_stage1_config"])
    relocated["dataset_path"] = "/relocated/cohort.parquet"
    relocated["architecture"]["htr_sentence_model"] = "/relocated/htr"
    relocated["architecture"]["agentic_feature_search"][
        "agent_server_url"
    ] = "http://relocated.invalid/v1"
    relocated["architecture"]["multi_model_forest"]["embedding_contrast"][
        "cache_dir"
    ] = "/relocated/cache"
    assert (
        artifact_module.stage1_effective_config_scientific_compatibility_projection(
            relocated
        )
        == baseline
    )

    changed = copy.deepcopy(relocated)
    changed["architecture"]["multi_model_forest"]["embedding_contrast"][
        "model_name"
    ] = "different-encoder"
    assert (
        artifact_module.stage1_effective_config_scientific_compatibility_projection(
            changed
        )
        != baseline
    )


def test_effective_config_projection_rejects_unaddressed_external_cache_paths():
    _audit, request = _fixture_values()
    changed = copy.deepcopy(request["effective_stage1_config"])
    changed["architecture"]["multi_model_forest"]["embedding_contrast"][
        "external_corpus_cache_dirs"
    ] = ["/unaddressed/external/corpus"]
    with pytest.raises(ValueError, match="external corpus path locators"):
        artifact_module.stage1_effective_config_scientific_compatibility_projection(
            changed
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "dataset",
        "config",
        "cache",
        "htr",
        "plan",
        "query",
        "producer",
        "security",
    ],
)
def test_preflight_artifact_rejects_scientific_request_substitution(
    tmp_path,
    identity_validators,
    mutation,
):
    audit, request = _fixture_values()
    substituted = copy.deepcopy(request)
    if mutation == "dataset":
        substituted["dataset"]["sha256"] = "a" * 64
    elif mutation == "config":
        substituted["effective_stage1_config"]["seed"] = 43
    elif mutation == "cache":
        substituted["embedding_cache"]["identity"]["sha256"] = "b" * 64
    elif mutation == "htr":
        substituted["htr_model"]["tree_sha256"] = "c" * 64
    elif mutation == "plan":
        substituted["stage1_scope_plan"]["scientific_content_sha256"] = "d" * 64
        _rehash_addressed(substituted["stage1_scope_plan"])
    elif mutation == "query":
        substituted["query_config"]["epochs"] = 121
    elif mutation == "producer":
        substituted["behavior_identity"]["source_tree_sha256"] = "e" * 64
        _rehash_addressed(substituted["behavior_identity"])
    else:
        substituted["security"]["oracle_accessed"] = True
    _rehash_request(substituted)
    artifact = _seal(tmp_path, audit, request)
    try:
        with pytest.raises(ValueError, match="scientific request differs"):
            _load(artifact, substituted)
    finally:
        _make_root_writable(artifact.root)
