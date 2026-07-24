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
    request_body = {
        "schema_version": "fixture_stage1_request",
        "dataset": {"path": "/fixture/data", "sha256": "1" * 64},
        "source_config": {"path": "/fixture/profile", "sha256": "2" * 64},
        "effective_stage1_config": {"seed": 42},
        "embedding_cache": {"identity": {"sha256": "3" * 64}},
        "htr_model": {"path": "/fixture/htr", "tree_sha256": "4" * 64},
        "split_registry_content_sha256": "5" * 64,
        "behavior_identity": _addressed({"source_tree_sha256": "6" * 64}),
        "hierarchical_discovery_contract_identity": _addressed(
            {"contract_sha256": "7" * 64}
        ),
        "architecture_contract": {"all_ten": True},
        "stage1_scope_plan": _addressed({"scope_order": audit["scope_order"]}),
        "query_config": {"epochs": 120},
        "runtime": {"preflight_workers": 8},
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


def test_preflight_artifact_rejects_a_substituted_stage1_request(
    tmp_path,
    identity_validators,
):
    audit, request = _fixture_values()
    substituted = copy.deepcopy(request)
    substituted["runtime"]["preflight_workers"] = 7
    substituted_body = {
        key: value
        for key, value in substituted.items()
        if key != "request_sha256"
    }
    substituted["request_sha256"] = artifact_module._sha256_json(substituted_body)
    artifact = _seal(tmp_path, audit, substituted)
    try:
        with pytest.raises(ValueError, match="differs"):
            _load(artifact, request)
    finally:
        _make_root_writable(artifact.root)
