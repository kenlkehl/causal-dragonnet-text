from __future__ import annotations

import json

import pytest

from oci.inference.all_evidence_discovery_interfaces import canonical_json, content_sha256
from oci.inference.hierarchical_all_architecture_discovery import (
    INTERPRET_CHUNK_JOB,
    DiscoveryJobSettings,
    DiscoveryJsonJob,
)
from oci.inference.hierarchical_discovery_response_contract import (
    attach_hierarchical_discovery_response_contract,
)
from oci.inference.hierarchical_discovery_job_cache import (
    HIERARCHICAL_DISCOVERY_JOB_CACHE_HIT_VERSION,
    AuthenticatedHierarchicalDiscoveryJobCache,
    HierarchicalDiscoveryJobCacheConfig,
)
from tests.hierarchy_resource_test_support import (
    HIERARCHY_JOB_CACHE_CONFIG,
)


def _runner_identity(*, label: str = "one") -> dict:
    body = {
        "schema_version": "closed_cache_test_runner_v1",
        "label": label,
        "endpoint_urls": ["http://offline.test/v1"],
        "model": {"name": "closed-json-model", "resolution": "explicit"},
        "retry": {"max_attempts": 1},
    }
    return {**body, "identity_sha256": content_sha256(body)}


def _job() -> DiscoveryJsonJob:
    request = attach_hierarchical_discovery_response_contract(
        job_kind=INTERPRET_CHUNK_JOB,
        request={
            "job": "interpret_evidence_chunk",
            "evidence": [
                {
                    "evidence_id": "evidence.cache_test",
                    "member_ids": [],
                }
            ],
        },
    )
    return DiscoveryJsonJob.create(
        job_kind=INTERPRET_CHUNK_JOB,
        scope="bow_nuisance.chunk_000",
        dependencies=(),
        settings=DiscoveryJobSettings.selector(),
        messages=(
            {"role": "system", "content": "Return the closed JSON object."},
            {"role": "user", "content": canonical_json(request)},
        ),
        input_bindings={"catalog_sha256": "a" * 64},
    )


def _validator(raw):
    if not isinstance(raw, dict) or set(raw) != {"answer"}:
        raise ValueError("semantic response has the wrong closed schema")
    answer = raw["answer"]
    if isinstance(answer, bool) or not isinstance(answer, int):
        raise TypeError("answer must be an integer")
    return {"answer": answer}


def _keyed_wire_validator(raw):
    if not isinstance(raw, dict) or set(raw) != {"by_id"}:
        raise ValueError("keyed wire response has the wrong closed schema")
    keyed = raw["by_id"]
    if not isinstance(keyed, dict) or set(keyed) != {"item.a", "item.b"}:
        raise ValueError("keyed wire response does not cover the exact identifiers")
    rows = []
    for identifier in ("item.a", "item.b"):
        value = keyed[identifier]
        if not isinstance(value, dict) or set(value) != {"value"}:
            raise ValueError("keyed wire row has the wrong closed schema")
        rows.append({"item_id": identifier, "value": value["value"]})
    return {"rows": rows}


def _begun_cache(tmp_path):
    root = tmp_path / "hierarchy_cache"
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    runner_identity = _runner_identity()
    inner_sha = "b" * 64
    validator_sha = "c" * 64
    cache.begin_execution(
        hierarchy_inner_precommit_sha256=inner_sha,
        runner_identity=runner_identity,
    )
    return cache, root, runner_identity, inner_sha, validator_sha


def _store(cache, runner_identity, inner_sha, validator_sha, *, answer=7):
    return cache.store_validated(
        job=_job(),
        hierarchy_inner_precommit_sha256=inner_sha,
        runner_identity=runner_identity,
        validator_code_sha256=validator_sha,
        validated_response={"answer": answer},
    )


def _entry_path(root):
    namespaces = tuple(root.iterdir())
    assert len(namespaces) == 1
    entries = tuple(namespaces[0].iterdir())
    assert len(entries) == 1
    return entries[0]


def test_identity_and_begin_execution_are_side_effect_free(tmp_path):
    root = tmp_path / "not_created_during_approval"
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=root,
        config=HierarchicalDiscoveryJobCacheConfig(max_entry_bytes=4096),
    )

    identity = cache.identity()
    cache.begin_execution(
        hierarchy_inner_precommit_sha256="b" * 64,
        runner_identity=_runner_identity(),
    )

    assert root.exists() is False
    assert identity["root_envelope"] == {
        "kind": "machine_local_absolute_path",
        "absolute_path": str(root),
    }
    assert identity["config"]["max_entry_bytes"] == 4096
    body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    assert identity["identity_sha256"] == content_sha256(body)


def test_validated_entry_is_immutable_and_replayed_through_same_validator(tmp_path):
    cache, root, runner_identity, inner_sha, validator_sha = _begun_cache(tmp_path)
    calls = []

    miss = cache.replay_validated(
        job=_job(),
        hierarchy_inner_precommit_sha256=inner_sha,
        runner_identity=runner_identity,
        validator_code_sha256=validator_sha,
        validator=lambda raw: calls.append(raw) or _validator(raw),
    )
    assert miss is None
    assert calls == []

    entry_sha = _store(cache, runner_identity, inner_sha, validator_sha)
    replay = cache.replay_validated(
        job=_job(),
        hierarchy_inner_precommit_sha256=inner_sha,
        runner_identity=runner_identity,
        validator_code_sha256=validator_sha,
        validator=lambda raw: calls.append(raw) or _validator(raw),
    )

    assert replay is not None
    assert replay.validated_response == {"answer": 7}
    assert replay.response_attempt_trace["mode"] == "single_validated_response"
    assert replay.response_attempt_trace["logical_job_id"] == _job().job_id
    assert calls == [{"answer": 7}]
    metadata = replay.execution_metadata
    assert metadata["schema_version"] == HIERARCHICAL_DISCOVERY_JOB_CACHE_HIT_VERSION
    assert metadata["record_type"] == "authenticated_cache_hit"
    assert metadata["cache_entry_sha256"] == entry_sha
    assert metadata["validated_response_sha256"] == content_sha256({"answer": 7})
    assert metadata["response_attempt_trace_sha256"] == content_sha256(
        replay.response_attempt_trace
    )
    record_body = {key: value for key, value in metadata.items() if key != "record_sha256"}
    assert metadata["record_sha256"] == content_sha256(record_body)
    assert len(cache.execution_metadata) == 1
    assert _entry_path(root).name == f"entry_{entry_sha}.json"

    assert _store(cache, runner_identity, inner_sha, validator_sha) == entry_sha
    with pytest.raises(ValueError, match="immutable cache entry differs"):
        _store(cache, runner_identity, inner_sha, validator_sha, answer=8)


def test_keyed_wire_and_deterministic_normalized_projection_are_bound_separately(tmp_path):
    cache, root, runner_identity, inner_sha, validator_sha = _begun_cache(tmp_path)
    wire = {
        "by_id": {
            "item.b": {"value": 2},
            "item.a": {"value": 1},
        }
    }
    normalized = _keyed_wire_validator(wire)
    assert content_sha256(wire) != content_sha256(normalized)

    cache.store_validated(
        job=_job(),
        hierarchy_inner_precommit_sha256=inner_sha,
        runner_identity=runner_identity,
        validator_code_sha256=validator_sha,
        wire_response=wire,
        validated_response=normalized,
    )
    entry = json.loads(_entry_path(root).read_text(encoding="utf-8"))
    assert entry["wire_response"] == wire
    assert entry["wire_response_sha256"] == content_sha256(wire)
    assert entry["validated_response"] == normalized
    assert entry["validated_response_sha256"] == content_sha256(normalized)
    assert entry["response_attempt_trace"]["wire_response_sha256"] == content_sha256(wire)
    assert entry["response_attempt_trace"]["validated_response_sha256"] == content_sha256(
        normalized
    )

    replay = cache.replay_validated(
        job=_job(),
        hierarchy_inner_precommit_sha256=inner_sha,
        runner_identity=runner_identity,
        validator_code_sha256=validator_sha,
        validator=_keyed_wire_validator,
    )
    assert replay is not None
    assert replay.wire_response == wire
    assert replay.validated_response == normalized
    assert replay.execution_metadata["wire_response_sha256"] == content_sha256(wire)
    assert replay.execution_metadata["validated_response_sha256"] == content_sha256(normalized)


@pytest.mark.parametrize(
    "tamper",
    ["bytes", "duplicate", "nonfinite", "extra", "identity", "response_trace"],
)
def test_tampered_cache_bytes_and_shapes_fail_closed(tmp_path, tamper):
    cache, root, runner_identity, inner_sha, validator_sha = _begun_cache(tmp_path)
    _store(cache, runner_identity, inner_sha, validator_sha)
    path = _entry_path(root)
    raw = path.read_bytes()
    entry = json.loads(raw)
    if tamper == "bytes":
        changed = raw + b"\n"
    elif tamper == "duplicate":
        changed = b'{"entry_sha256":"' + b"0" * 64 + b'",' + raw[1:]
    elif tamper == "nonfinite":
        entry["validated_response"]["answer"] = float("nan")
        changed = json.dumps(
            entry,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=True,
            separators=(",", ":"),
        ).encode()
    elif tamper == "extra":
        entry["unexpected"] = True
        changed = canonical_json(entry).encode()
    elif tamper == "identity":
        entry["lookup_identity"]["runner_identity"]["label"] = "forged"
        changed = canonical_json(entry).encode()
    else:
        entry["response_attempt_trace"]["logical_job_id"] = "job_" + "0" * 64
        changed = canonical_json(entry).encode()
    path.write_bytes(changed)

    with pytest.raises((TypeError, ValueError)):
        cache.replay_validated(
            job=_job(),
            hierarchy_inner_precommit_sha256=inner_sha,
            runner_identity=runner_identity,
            validator_code_sha256=validator_sha,
            validator=_validator,
        )


def test_symlink_or_unexpected_namespace_entry_fails_closed(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    symlink_root = tmp_path / "cache_link"
    symlink_root.symlink_to(target, target_is_directory=True)
    cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=symlink_root,
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    cache.begin_execution(
        hierarchy_inner_precommit_sha256="b" * 64,
        runner_identity=_runner_identity(),
    )
    with pytest.raises(ValueError, match="root cannot be a symlink"):
        cache.replay_validated(
            job=_job(),
            hierarchy_inner_precommit_sha256="b" * 64,
            runner_identity=_runner_identity(),
            validator_code_sha256="c" * 64,
            validator=_validator,
        )

    cache, root, runner_identity, inner_sha, validator_sha = _begun_cache(tmp_path / "second")
    _store(cache, runner_identity, inner_sha, validator_sha)
    namespace = _entry_path(root).parent
    (namespace / "unexpected.txt").write_text("not admitted", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected entry"):
        cache.replay_validated(
            job=_job(),
            hierarchy_inner_precommit_sha256=inner_sha,
            runner_identity=runner_identity,
            validator_code_sha256=validator_sha,
            validator=_validator,
        )


def test_changed_runner_or_validator_identity_cannot_hit_existing_entry(tmp_path):
    cache, _root, runner_identity, inner_sha, validator_sha = _begun_cache(tmp_path)
    _store(cache, runner_identity, inner_sha, validator_sha)

    changed_runner = _runner_identity(label="two")
    cache.begin_execution(
        hierarchy_inner_precommit_sha256=inner_sha,
        runner_identity=changed_runner,
    )
    assert (
        cache.replay_validated(
            job=_job(),
            hierarchy_inner_precommit_sha256=inner_sha,
            runner_identity=changed_runner,
            validator_code_sha256=validator_sha,
            validator=_validator,
        )
        is None
    )

    cache.begin_execution(
        hierarchy_inner_precommit_sha256=inner_sha,
        runner_identity=runner_identity,
    )
    assert (
        cache.replay_validated(
            job=_job(),
            hierarchy_inner_precommit_sha256=inner_sha,
            runner_identity=runner_identity,
            validator_code_sha256="d" * 64,
            validator=_validator,
        )
        is None
    )
