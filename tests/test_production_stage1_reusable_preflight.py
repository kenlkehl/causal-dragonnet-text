from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Callable

import pytest
import numpy as np

import oci.inference.embedding_contrast_discovery as embedding_discovery
import oci.inference.prepared_stage1_context as prepared_context_module
import oci.inference.production_stage1_bundle as bundle_module
import oci.inference.production_stage1_cluster_preflight_artifact as legacy_preflight_module
import oci.inference.production_stage1_cluster_preflight_artifact_v2 as portable_preflight_module
import oci.inference.production_stage1_reusable_preflight as reusable
from oci.inference.production_stage1_bundle import (
    _embedding_cache_cluster_preflight_scientific_selector,
)
from oci.inference.production_stage1_scope_scheduler import (
    Stage1PhysicalFitIdentity,
    build_canonical_stage1_scope_plan,
)
from tests.stage1_test_support import PHYSICAL_FIT_IDENTITY
from tests.test_production_stage1_role_neutral_execution import (
    _registry,
)
from tests.test_prepared_stage1_context import (
    _options as _prepared_context_options,
)
from tests.test_role_neutral_embedding_group_execution import (
    _one_physical_group_plan,
    _preflight_and_states,
    _request as _embedding_request,
    _texts as _embedding_texts,
    _write_cache as _write_embedding_cache,
)


def _sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _global_case(tmp_path: Path) -> tuple[Path, dict[str, object], str]:
    store = (tmp_path / "reusable_store").resolve()
    compatibility: dict[str, object] = {
        "schema_version": reusable.GLOBAL_AUDIT_COMPATIBILITY_SCHEMA,
        "prepared_cohort_content_sha256": _sha(
            ["note zero", "note one"]
        ),
        "ordered_row_identity_sha256": _sha([0, 1]),
        "htr_model_tree_sha256": "1" * 64,
        "tokenizer_identity_sha256": "2" * 64,
        "chunking_configuration_sha256": "3" * 64,
        "producer_schema_identity": (
            "test_global_nontruncation_producer_v1"
        ),
    }
    chunk_counts = [2, 1]
    token_lengths = [3, 4, 5]
    audit_body = {
        "schema_version": "test_exact_nontruncation_audit_v1",
        "row_count": 2,
        "total_chunks": 3,
        "normalized_text_projection_sha256": _sha(
            ["note zero", "note one"]
        ),
        "ordered_chunk_counts_sha256": _sha(chunk_counts),
        "ordered_token_counts_sha256": _sha(token_lengths),
        "htr_model_tree_sha256": "1" * 64,
        "chunk_cap_nonbinding": True,
        "all_chunks_within_effective_max_length": True,
        "semantic_truncation_allowed": False,
        "tokenizer_truncation_allowed": False,
    }
    audit = {**audit_body, "content_sha256": _sha(audit_body)}
    producer_identity = "test_global_nontruncation_producer_v1"
    reusable.seal_reusable_global_audit(
        store_root=store,
        compatibility=compatibility,
        audit=audit,
        row_text_sha256=(
            hashlib.sha256(b"note zero").hexdigest(),
            hashlib.sha256(b"note one").hexdigest(),
        ),
        row_chunk_counts=chunk_counts,
        token_lengths=token_lengths,
        producer_identity=producer_identity,
    )
    return store, compatibility, producer_identity


def _load_global(
    store: Path,
    compatibility: dict[str, object],
    producer_identity: str,
) -> reusable.ReusableGlobalAuditArtifact:
    return reusable.load_reusable_global_audit(
        store_root=store,
        compatibility=compatibility,
        producer_identity=producer_identity,
    )


def _global_root(
    store: Path,
    compatibility: dict[str, object],
) -> Path:
    key = reusable.scientific_key(
        compatibility,
        expected_schema=reusable.GLOBAL_AUDIT_COMPATIBILITY_SCHEMA,
    )
    return store / "global_audits" / key


def _scope_plan(
    *,
    gpu_ids: tuple[int, ...] = (0,),
    scope_workers_per_gpu: int = 1,
    global_seed: int = 42,
    registry_sha256: str = "a" * 64,
    physical_fit_identity: Stage1PhysicalFitIdentity = (
        PHYSICAL_FIT_IDENTITY
    ),
):
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256=registry_sha256,
        global_seed=global_seed,
        physical_fit_identity=physical_fit_identity,
        gpu_ids=gpu_ids,
        review_rounds=2,
        initial_training_partitions=3,
        scope_workers_per_gpu=scope_workers_per_gpu,
        expected_outer_fold_count=5,
        expected_inner_fold_count=5,
    )


def test_cluster_cache_selector_ignores_relocation_mechanism() -> None:
    build = {
        "schema_version": "fixture_cache_build_v1",
        "dataset_sha256": "1" * 64,
        "ordered_text_sha256": "2" * 64,
        "sentence_model_name": "fixture/encoder",
        "local_model_tree_sha256": "3" * 64,
        "chunk_configuration_sha256": "4" * 64,
        "cache_configuration_sha256": "5" * 64,
        "row_count": 1000,
        "chunk_count": 2000,
        "hidden_size": 64,
        "cache_files": {
            "chunk_embeddings.npy": {
                "sha256": "6" * 64,
                "size_bytes": 123,
            }
        },
        "provider_identity": {
            "provider": "fixture_spent_cache_v1",
            "embeddings_sha256": "6" * 64,
        },
    }
    relocated = {
        "schema_version": "fixture_relocation_v9",
        "relocator_code_sha256": "7" * 64,
        "root": "/another/mount/cache",
        "cache_build_identity": build,
    }
    projected_request = {
        "identity": build["provider_identity"],
        "production_cache_build_identity": build,
        "authenticated_relocation": relocated,
        "legacy_terminal_migration_identity": {
            "producer": "unrelated adoption mechanism"
        },
    }

    expected = (
        _embedding_cache_cluster_preflight_scientific_selector(build)
    )
    assert (
        _embedding_cache_cluster_preflight_scientific_selector(
            relocated
        )
        == expected
    )
    assert (
        _embedding_cache_cluster_preflight_scientific_selector(
            projected_request
        )
        == expected
    )
    changed = copy.deepcopy(build)
    changed["cache_files"]["chunk_embeddings.npy"]["sha256"] = (
        "8" * 64
    )
    assert (
        _embedding_cache_cluster_preflight_scientific_selector(
            changed
        )
        != expected
    )


def test_hash_only_legacy_global_audit_cannot_claim_complete_inventory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="materialize exact row/chunk arrays",
    ):
        reusable.seal_reusable_global_audit_from_authenticated_legacy(
            store_root=(tmp_path / "legacy-global-store").resolve(),
            compatibility={},
            audit={},
            authenticated_source={},
            producer_identity="legacy-fixture",
        )
    assert not (tmp_path / "legacy-global-store").exists()


def test_preflight_plan_projection_excludes_resources_and_broad_fit_identity(
) -> None:
    baseline = reusable.preflight_scope_plan_projection(_scope_plan())
    unrelated_fit_identity = Stage1PhysicalFitIdentity(
        architecture_identity="4" * 64,
        target=PHYSICAL_FIT_IDENTITY.target,
        scientific_configuration_identity="5" * 64,
        producer_identity="6" * 64,
        runtime_compatibility_class=(
            PHYSICAL_FIT_IDENTITY.runtime_compatibility_class
        ),
    )
    operationally_changed = reusable.preflight_scope_plan_projection(
        _scope_plan(
            gpu_ids=(3, 7),
            scope_workers_per_gpu=4,
            physical_fit_identity=unrelated_fit_identity,
        )
    )

    assert operationally_changed == baseline
    assert baseline["physical_fit_identity_included"] is False
    assert baseline["resource_assignment_included"] is False


@pytest.mark.parametrize(
    "changed",
    (
        _scope_plan(global_seed=43),
        _scope_plan(registry_sha256="b" * 64),
    ),
    ids=("seed", "split-registry"),
)
def test_preflight_plan_projection_binds_seed_and_split(
    changed,
) -> None:
    assert reusable.preflight_scope_plan_projection(
        changed
    ) != reusable.preflight_scope_plan_projection(_scope_plan())


def test_reusable_global_audit_first_authenticates_bytes_then_reopens_by_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, compatibility, producer = _global_case(tmp_path)
    first = _load_global(store, compatibility, producer)
    assert first.authentication_mode == "prior_proof_stat_continuity"

    payload_hash_reads: list[str] = []
    original = reusable._stable_file_sha256

    def traced(path: Path) -> tuple[str, int]:
        resolved = Path(path)
        if resolved.name in {
            "row_chunk_counts.npy",
            "chunk_token_lengths.npy",
        }:
            payload_hash_reads.append(resolved.name)
        return original(resolved)

    monkeypatch.setattr(reusable, "_stable_file_sha256", traced)
    reopened = _load_global(store, compatibility, producer)

    assert reopened.authentication_mode == "prior_proof_stat_continuity"
    assert payload_hash_reads == []
    assert reopened.payload_bytes_read < sum(
        path.stat().st_size
        for path in _global_root(store, compatibility).iterdir()
        if path.is_file()
    )


def test_stat_discontinuity_falls_back_to_deep_authentication(
    tmp_path: Path,
) -> None:
    store, compatibility, producer = _global_case(tmp_path)
    root = _global_root(store, compatibility)
    payload = root / "row_chunk_counts.npy"
    before = payload.stat()
    os.utime(
        payload,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
    )

    reopened = _load_global(store, compatibility, producer)
    assert reopened.authentication_mode == "full_byte_reauthentication"
    assert reopened.payload_bytes_read >= payload.stat().st_size

    continuous = _load_global(store, compatibility, producer)
    assert continuous.authentication_mode == "prior_proof_stat_continuity"


def test_unprotected_proof_ancestry_cannot_use_fast_path(
    tmp_path: Path,
) -> None:
    store, compatibility, producer = _global_case(tmp_path)
    key = reusable.scientific_key(
        compatibility,
        expected_schema=reusable.GLOBAL_AUDIT_COMPATIBILITY_SCHEMA,
    )
    kind_root = (
        store / "authentication_proofs" / "global_audit"
    )
    kind_root.chmod(0o770)

    reopened = _load_global(store, compatibility, producer)

    assert reopened.authentication_mode == "full_byte_reauthentication"
    assert reopened.payload_bytes_read > 0
    assert stat.S_IMODE(kind_root.stat().st_mode) == 0o700
    assert (
        kind_root
        / key
        / "authentication_receipt_terminal.json"
    ).is_file()


def test_public_try_load_quarantines_invalid_payload_then_allows_recompute(
    tmp_path: Path,
) -> None:
    store, compatibility, producer = _global_case(tmp_path)
    root = _global_root(store, compatibility)
    payload = root / "row_chunk_counts.npy"
    payload.chmod(0o600)
    changed = bytearray(payload.read_bytes())
    changed[-1] ^= 1
    payload.write_bytes(changed)
    payload.chmod(0o444)

    assert (
        reusable.try_load_reusable_global_audit(
            store_root=store,
            compatibility=compatibility,
            producer_identity=producer,
        )
        is None
    )
    assert not root.exists()
    recovery = store / "recovery" / "global_audits"
    quarantined = tuple(
        path for path in recovery.iterdir() if path.is_dir()
    )
    records = tuple(
        path
        for path in recovery.iterdir()
        if path.name.endswith(".recovery.json")
    )
    assert len(quarantined) == 1
    assert len(records) == 1

    rebuilt_store, rebuilt_compatibility, rebuilt_producer = (
        _global_case(tmp_path)
    )
    rebuilt = _load_global(
        rebuilt_store,
        rebuilt_compatibility,
        rebuilt_producer,
    )
    assert rebuilt.scientific_key == reusable.scientific_key(
        compatibility,
        expected_schema=reusable.GLOBAL_AUDIT_COMPATIBILITY_SCHEMA,
    )
    assert rebuilt.authentication_mode == "prior_proof_stat_continuity"


def test_byte_identical_inode_replacement_cannot_use_fast_path(
    tmp_path: Path,
) -> None:
    store, compatibility, producer = _global_case(tmp_path)
    root = _global_root(store, compatibility)
    payload = root / "chunk_token_lengths.npy"
    original_inode = payload.stat().st_ino
    replacement = tmp_path / "replacement.npy"
    replacement.write_bytes(payload.read_bytes())
    replacement.chmod(0o444)
    root.chmod(0o755)
    os.replace(replacement, payload)
    root.chmod(0o555)
    assert payload.stat().st_ino != original_inode

    reopened = _load_global(store, compatibility, producer)
    assert reopened.authentication_mode == "full_byte_reauthentication"


def test_missing_authentication_proof_falls_back_to_deep_authentication(
    tmp_path: Path,
) -> None:
    store, compatibility, producer = _global_case(tmp_path)
    key = reusable.scientific_key(
        compatibility,
        expected_schema=reusable.GLOBAL_AUDIT_COMPATIBILITY_SCHEMA,
    )
    proof = (
        store
        / "authentication_proofs"
        / "global_audit"
        / key
        / "authentication_proof.json"
    )
    proof.unlink()

    reopened = _load_global(store, compatibility, producer)
    assert reopened.authentication_mode == "full_byte_reauthentication"
    assert proof.is_file()


def test_optional_proof_establishment_failure_keeps_valid_deep_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, compatibility, producer = _global_case(tmp_path)
    key = reusable.scientific_key(
        compatibility,
        expected_schema=reusable.GLOBAL_AUDIT_COMPATIBILITY_SCHEMA,
    )
    proof_root = (
        store / "authentication_proofs" / "global_audit" / key
    )
    shutil.rmtree(proof_root)

    def unavailable(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError(
            "filesystem timestamp barrier is unavailable"
        )

    monkeypatch.setattr(
        reusable,
        "_publish_full_auth_proof",
        unavailable,
    )
    reopened = _load_global(store, compatibility, producer)

    assert reopened.authentication_mode == "full_byte_reauthentication"
    assert _global_root(store, compatibility).is_dir()
    assert not proof_root.exists()
    assert (
        reusable.try_load_reusable_global_audit(
            store_root=store,
            compatibility=compatibility,
            producer_identity=producer,
        )
        is not None
    )


def test_relocated_store_keeps_scientific_key_and_reestablishes_local_proof(
    tmp_path: Path,
) -> None:
    store, compatibility, producer = _global_case(tmp_path)
    relocated = (tmp_path / "relocated_store").resolve()
    shutil.copytree(store, relocated, copy_function=shutil.copy2)

    reopened = _load_global(relocated, compatibility, producer)
    assert reopened.scientific_key == reusable.scientific_key(
        compatibility,
        expected_schema=reusable.GLOBAL_AUDIT_COMPATIBILITY_SCHEMA,
    )
    assert reopened.authentication_mode == "full_byte_reauthentication"

    continuous = _load_global(relocated, compatibility, producer)
    assert continuous.authentication_mode == "prior_proof_stat_continuity"


def test_same_tick_terminal_byte_mutation_is_never_accepted(
    tmp_path: Path,
) -> None:
    store, compatibility, producer = _global_case(tmp_path)
    terminal = (
        _global_root(store, compatibility) / reusable.GLOBAL_TERMINAL
    )
    before = terminal.stat()
    terminal.chmod(0o600)
    payload = bytearray(terminal.read_bytes())
    payload[payload.index(b"test_global") + 5] ^= 1
    terminal.write_bytes(payload)
    terminal.chmod(0o444)
    after = terminal.stat()
    # /data1 has coarse enough timestamps that an immediate same-size
    # rewrite can retain the entire stat identity. The terminal's own
    # authenticated bytes must still reject it.
    assert before.st_size == after.st_size

    with pytest.raises((RuntimeError, ValueError)):
        _load_global(store, compatibility, producer)


def _alter_bytes(root: Path, payload: Path, tmp_path: Path) -> None:
    del root, tmp_path
    payload.chmod(0o600)
    value = bytearray(payload.read_bytes())
    value[-1] ^= 1
    payload.write_bytes(value)
    payload.chmod(0o444)


def _remove_payload(root: Path, payload: Path, tmp_path: Path) -> None:
    del tmp_path
    root.chmod(0o755)
    payload.unlink()
    root.chmod(0o555)


def _add_duplicate(root: Path, payload: Path, tmp_path: Path) -> None:
    del tmp_path
    root.chmod(0o755)
    duplicate = root / "unregistered_duplicate.npy"
    duplicate.write_bytes(payload.read_bytes())
    duplicate.chmod(0o444)
    root.chmod(0o555)


def _replace_with_symlink(
    root: Path,
    payload: Path,
    tmp_path: Path,
) -> None:
    external = tmp_path / "external.npy"
    external.write_bytes(payload.read_bytes())
    root.chmod(0o755)
    payload.unlink()
    payload.symlink_to(external)
    root.chmod(0o555)


def _replace_with_hardlink(
    root: Path,
    payload: Path,
    tmp_path: Path,
) -> None:
    external = tmp_path / "external.npy"
    external.write_bytes(payload.read_bytes())
    external.chmod(0o444)
    root.chmod(0o755)
    payload.unlink()
    os.link(external, payload)
    root.chmod(0o555)
    assert payload.stat().st_nlink == 2


def _alter_permissions(
    root: Path,
    payload: Path,
    tmp_path: Path,
) -> None:
    del root, tmp_path
    payload.chmod(0o644)


@pytest.mark.parametrize(
    "mutate",
    [
        _alter_bytes,
        _remove_payload,
        _add_duplicate,
        _replace_with_symlink,
        _replace_with_hardlink,
        _alter_permissions,
    ],
    ids=(
        "altered-bytes",
        "missing",
        "duplicated",
        "symlinked",
        "hard-linked",
        "permission-altered",
    ),
)
def test_invalid_payload_tree_neither_uses_fast_path_nor_authenticates_deeply(
    tmp_path: Path,
    mutate: Callable[[Path, Path, Path], None],
) -> None:
    store, compatibility, producer = _global_case(tmp_path)
    root = _global_root(store, compatibility)
    payload = root / "chunk_token_lengths.npy"
    mutate(root, payload, tmp_path)

    with pytest.raises(
        (FileNotFoundError, OSError, RuntimeError, ValueError)
    ):
        _load_global(store, compatibility, producer)


def test_fast_proof_records_complete_stat_identity_and_scientific_bindings(
    tmp_path: Path,
) -> None:
    store, compatibility, _producer = _global_case(tmp_path)
    key = reusable.scientific_key(
        compatibility,
        expected_schema=reusable.GLOBAL_AUDIT_COMPATIBILITY_SCHEMA,
    )
    proof_root = (
        store / "authentication_proofs" / "global_audit" / key
    )
    proof = json.loads(
        (proof_root / "authentication_proof.json").read_text(
            encoding="utf-8"
        )
    )
    terminal = json.loads(
        (
            proof_root / "authentication_receipt_terminal.json"
        ).read_text(encoding="utf-8")
    )
    required_stat_fields = {
        "device",
        "inode",
        "mode",
        "link_count",
        "uid",
        "gid",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
    }

    assert proof["ordinary_full_byte_authentication_completed"] is True
    assert proof["path_in_scientific_identity"] is False
    assert proof["mtime_only_trust_used"] is False
    assert proof["timestamp_stabilization_protocol"] == (
        "same_filesystem_probe_then_vulnerable_byte_recheck_v1"
    )
    assert (
        proof["artifact_barrier_probe_ctime_ns"]
        > proof["full_authentication_start_probe_ctime_ns"]
    )
    assert proof["tree_stat_inventory"]
    assert reusable.GLOBAL_TERMINAL in {
        row["relative_path"] for row in proof["tree_stat_inventory"]
    }
    assert all(
        required_stat_fields <= set(row)
        for row in proof["tree_stat_inventory"]
    )
    assert terminal["ordinary_full_byte_authentication_completed"] is True
    assert terminal["timestamp_stabilization_barrier_completed"] is True
    assert (
        terminal["authentication_proof_registration"]["stat_identity"].keys()
        == required_stat_fields
    )
    assert stat.S_IMODE(
        (proof_root / "authentication_proof.json").stat().st_mode
    ) == 0o600
    assert stat.S_IMODE(proof_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(proof_root.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(proof_root.parent.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("artifact_kind", "schema_identity"),
    (
        ("global_audit", reusable.REUSABLE_GLOBAL_AUDIT_SCHEMA),
        ("owner_artifact", reusable.REUSABLE_OWNER_ARTIFACT_SCHEMA),
        (
            "assembled_context",
            reusable.REUSABLE_ASSEMBLED_ARTIFACT_SCHEMA,
        ),
        (
            "accepted_context",
            reusable.REUSABLE_ACCEPTED_CONTEXT_SCHEMA,
        ),
    ),
)
def test_full_authentication_rejects_terminal_mutation_during_stat_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
    schema_identity: str,
) -> None:
    """No artifact kind may seal a proof around an in-flight terminal edit."""

    store = (tmp_path / "store").resolve()
    store.mkdir()
    root = (tmp_path / f"{artifact_kind}_artifact").resolve()
    root.mkdir()
    terminal = root / "terminal.json"
    original = b'{"status":"complete","sentinel":"alpha"}\n'
    altered = b'{"status":"complete","sentinel":"omega"}\n'
    assert len(original) == len(altered)
    terminal.write_bytes(original)
    terminal.chmod(0o444)
    root.chmod(0o555)

    digest = hashlib.sha256(original).hexdigest()
    start_probe = reusable._authentication_probe_ctime_ns(store)
    mutated = False

    def mutate_before_barrier_completes(
        *,
        store_root: Path,
        after_ctime_ns: int,
        timeout_seconds: float = 10.0,
    ) -> int:
        nonlocal mutated
        del store_root, timeout_seconds
        if not mutated:
            root.chmod(0o755)
            terminal.chmod(0o644)
            terminal.write_bytes(altered)
            terminal.chmod(0o444)
            root.chmod(0o555)
            mutated = True
        return int(after_ctime_ns) + 1

    monkeypatch.setattr(
        reusable,
        "_wait_for_timestamp_barrier",
        mutate_before_barrier_completes,
    )
    with pytest.raises(RuntimeError, match="changed"):
        reusable._publish_full_auth_proof(
            store_root=store,
            artifact_kind=artifact_kind,
            scientific_key="a" * 64,
            artifact_root=root,
            terminal_content_sha256=digest,
            artifact_scientific_content_sha256="b" * 64,
            producer_identity="test_producer_v1",
            schema_identity=schema_identity,
            full_authentication_start_probe_ctime_ns=start_probe,
            authenticated_byte_inventory={
                "terminal.json": (digest, len(original)),
            },
        )
    assert mutated is True
    assert not (
        store
        / "authentication_proofs"
        / artifact_kind
        / ("a" * 64)
    ).exists()


def _complete_one_owner_cluster_audit(
    tmp_path: Path,
) -> tuple[
    object,
    dict[str, object],
    str,
    dict[str, object],
]:
    """Return a complete logical audit plus one already-fitted owner state."""

    plan = _one_physical_group_plan()
    request = _embedding_request(plan)
    texts = _embedding_texts()
    cache = _write_embedding_cache(
        tmp_path / "fixture_embedding_cache",
        texts=texts,
    )
    legacy, _sealed_state, kmeans, svds = _preflight_and_states(
        tmp_path=tmp_path / "fixture_legacy_preflight",
        request=request,
        cache=cache,
    )
    source_audit = copy.deepcopy(dict(legacy.audit))
    source_scope = source_audit["scopes"][0]
    fit_identity = source_scope["cluster_fit_identity"]
    family = "cluster_local_treatment_contrast_basis"
    fit_identity["raw_cluster_concepts"] = [
        {"contrast_family": family, "complete": True}
    ]
    fit_identity["raw_cluster_concepts_sha256"] = _sha(
        fit_identity["raw_cluster_concepts"]
    )
    fit_identity["semantic_cluster_concepts"] = [
        {"contrast_family": family, "complete": True}
    ]
    fit_identity["semantic_cluster_concepts_sha256"] = _sha(
        fit_identity["semantic_cluster_concepts"]
    )
    fit_body = {
        key: child
        for key, child in fit_identity.items()
        if key != "content_sha256"
    }
    fit_identity["content_sha256"] = _sha(fit_body)
    owner = request.physical_owner.scope_id
    complete_scopes = []
    for logical_scope in plan.scopes:
        scope = copy.deepcopy(source_scope)
        scope.update(
            {
                "scope_id": logical_scope.scope_id,
                "scope_kind": logical_scope.scope_kind,
                "outer_fold": logical_scope.outer_fold,
                "inner_fold": logical_scope.inner_fold,
                "context_epoch": logical_scope.context_epoch,
                "provider_inner_fold": (
                    logical_scope.provider_inner_fold
                ),
                "fit_row_count": logical_scope.fit_row_count,
                "fit_row_order_fingerprint": _sha(
                    list(logical_scope.fit_row_ids)
                ),
                "canonical_group_seed": logical_scope.scope_seed,
                "heldout_row_count": logical_scope.heldout_row_count,
                "heldout_row_order_fingerprint": _sha(
                    list(logical_scope.heldout_row_ids)
                ),
            }
        )
        binding_body = {
            "schema_version": (
                "production_stage1_cluster_preflight_physical_binding_v2"
            ),
            "logical_scope_id": logical_scope.scope_id,
            "physical_owner_scope_id": owner,
            "reuses_physical_fit": logical_scope.scope_id != owner,
        }
        scope["physical_fit_binding"] = {
            **binding_body,
            "content_sha256": _sha(binding_body),
        }
        complete_scopes.append(scope)
    source_body = {
        **{
            key: child
            for key, child in source_audit.items()
            if key != "content_sha256"
        },
        "scope_order": [
            logical_scope.scope_id for logical_scope in plan.scopes
        ],
        "scopes": complete_scopes,
        "physical_fit_count": 1,
        "deduplicated_fit_count": len(plan.scopes) - 1,
        "physical_scope_order": [owner],
        "all_required_scopes_passed": True,
    }
    complete_audit = {
        **source_body,
        "content_sha256": _sha(source_body),
    }
    captured_state = {
        "schema_version": (
            "production_stage1_cluster_preflight_scope_state_capture_v2"
        ),
        "scope_id": owner,
        "cluster_fit_identity_content_sha256": fit_identity[
            "content_sha256"
        ],
        "kmeans_state": kmeans,
        "svd_states": svds,
        "captured_from_canonical_preflight_fit": True,
        "refit_performed_for_state_capture": False,
    }
    return plan, complete_audit, owner, captured_state


def test_cold_preflight_acceptance_fast_reopen_reads_no_bulk_or_fit_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unchanged accepted context reopens by stat continuity only."""

    cold_started = time.perf_counter()
    store, global_compatibility, global_producer = _global_case(tmp_path)
    global_audit = _load_global(
        store,
        global_compatibility,
        global_producer,
    )
    plan, audit, owner, captured_state = (
        _complete_one_owner_cluster_audit(tmp_path)
    )
    reference = (
        portable_preflight_module.build_portable_cluster_preflight_reference(
            audit
        )
    )
    request_body = {
        "schema_version": "fixture_reusable_stage1_request_v1",
        "split_registry_content_sha256": (
            plan.registry_content_sha256
        ),
        "stage1_scope_plan": plan.as_dict(),
        "embedding_cluster_feasibility_audit": reference,
        "htr_input_nontruncation_audit": {
            "content_sha256": global_audit.audit["content_sha256"],
        },
        "semantic_witness_scientific_config": {
            "schema_version": "fixture_semantic_witness_v1",
            "lossless": True,
        },
    }
    stage1_request = {
        **request_body,
        "request_sha256": _sha(request_body),
    }

    def validate_fixture_request(
        request: object,
        *,
        expected_reference: object,
    ) -> dict[str, object]:
        value = copy.deepcopy(dict(request))
        assert value["embedding_cluster_feasibility_audit"] == (
            expected_reference
        )
        assert value["request_sha256"] == _sha(
            {
                key: child
                for key, child in value.items()
                if key != "request_sha256"
            }
        )
        return value

    monkeypatch.setattr(
        reusable,
        "_validate_stage1_request_with_reference",
        validate_fixture_request,
    )
    cluster_compatibility = {
        "schema_version": "fixture_cluster_compatibility_v1",
        "embedding_cache_scientific_sha256": "7" * 64,
        "cluster_configuration_sha256": "8" * 64,
        "producer_schema_identity": "fixture_cluster_producer_v1",
    }
    physical_scope = plan.physical_scopes[0]
    fit_binding = {
        "schema_version": (
            "production_stage1_owner_fit_input_binding_v2"
        ),
        "ordered_fit_modeling_rows_sha256": "9" * 64,
        "ordered_fit_embedding_rows_sha256": "a" * 64,
        "ordered_fit_row_count": physical_scope.fit_row_count,
        "embedding_row_digest_schema_version": (
            "fixture_embedding_row_digest_v1"
        ),
    }
    owner_compatibility = reusable.owner_compatibility(
        cluster_compatibility=cluster_compatibility,
        physical_scope=physical_scope.as_dict(),
        fit_input_binding=fit_binding,
    )
    scope_audit = next(
        scope
        for scope in audit["scopes"]
        if scope["scope_id"] == owner
    )
    owner_producer = "fixture_reusable_owner_producer_v1"
    owner_handle = reusable.seal_reusable_owner_artifact(
        store_root=store,
        compatibility=owner_compatibility,
        scope_audit=scope_audit,
        captured_state=captured_state,
        producer_identity=owner_producer,
        parquet_compression="none",
    )
    owner_key = owner_handle.scientific_key
    assembled_compatibility = reusable.assembled_compatibility(
        cluster_compatibility=cluster_compatibility,
        preflight_plan_content_sha256=(
            reusable.preflight_scope_plan_projection(plan)[
                "content_sha256"
            ]
        ),
        physical_owner_keys={owner: owner_key},
        global_audit_scientific_key=global_audit.scientific_key,
    )
    assembled_producer = "fixture_reusable_assembled_producer_v1"
    assembled = reusable.seal_reusable_assembled_preflight(
        store_root=store,
        compatibility=assembled_compatibility,
        audit=audit,
        stage1_request=stage1_request,
        owner_handles={owner: owner_handle},
        global_audit=global_audit,
        plan=plan,
        producer_identity=assembled_producer,
        owner_producer_identity=owner_producer,
        global_audit_producer_identity=global_producer,
    )

    projection_body = {
        "schema_version": "fixture_preflight_projection_v1",
        "split_registry_content_sha256": (
            plan.registry_content_sha256
        ),
        "request_scientific_sha256": stage1_request[
            "request_sha256"
        ],
    }
    projection = {
        **projection_body,
        "content_sha256": _sha(projection_body),
    }
    monkeypatch.setattr(
        legacy_preflight_module,
        "stage1_request_scientific_compatibility_projection",
        lambda _request: copy.deepcopy(projection),
    )
    architecture_profiles = {"fixture": {"closed": True}}
    runtime_class = "fixture-reusable-preflight-runtime-v1"
    registry = {
        "schema_version": "fixture_split_registry_v1",
        "dataset_row_count": len(_embedding_texts()),
    }
    options = _prepared_context_options(tmp_path, "accepted-source")
    profile_path = Path(str(options["config_path"]))
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        '{"schema_version":"fixture_stage1_profile_v1"}\n',
        encoding="utf-8",
    )
    scientific = prepared_context_module._scientific_payload(
        stage1_request,
        registry=registry,
        registry_content_sha256=plan.registry_content_sha256,
        architecture_profiles=architecture_profiles,
        runtime_compatibility_class=runtime_class,
    )
    locator = prepared_context_module._locator_payload(
        stage1_build_options=options,
        architecture_profiles=architecture_profiles,
        runtime_compatibility_class=runtime_class,
        scientific_compatibility_sha256=scientific[
            "stage1_request_scientific_compatibility_sha256"
        ],
        exact_stage1_request=stage1_request,
    )
    prepared = (
        prepared_context_module._publish_prepared_stage1_context_payloads(
            root=(tmp_path / "prepared_context").resolve(),
            scientific=scientific,
            locator=locator,
        )
    )
    selector_body = {
        "schema_version": (
            "production_stage1_preflight_accepted_input_selector_v2"
        ),
        "prepared_scientific_content_sha256": (
            prepared.content_root_sha256
        ),
        "assembled_scientific_content_sha256": assembled.identity()[
            "path_neutral_scientific_content_sha256"
        ],
        "operational_paths_included": False,
        "stage2_identity_included": False,
    }
    selector = {
        **selector_body,
        "content_sha256": _sha(selector_body),
    }
    first = reusable.publish_reusable_preflight_acceptance(
        store_root=store,
        selector=selector,
        artifact=assembled,
        prepared_context_manifest_path=prepared.manifest_path,
        producer_identity=assembled_producer,
        owner_producer_identity=owner_producer,
        global_audit_producer_identity=global_producer,
    )
    first_identity = first.preflight.identity()
    first_state_identity = first.state_bundle.content_sha256
    first_context_identity = first.prepared_context.content_root_sha256
    assert first.preflight.authentication[
        "owner_state_payloads_deserialized"
    ] == 0
    cold_elapsed = time.perf_counter() - cold_started
    cold_payload_bytes = sum(
        path.stat().st_size
        for kind in (
            "global_audits",
            "owner_artifacts",
            "assembled_contexts",
            "accepted_contexts",
        )
        for path in (store / kind).rglob("*")
        if path.is_file() and not path.is_symlink()
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "unchanged reusable preflight repeated scientific work"
        )

    monkeypatch.setattr(
        bundle_module,
        "_load_local_htr_tokenizer",
        forbidden,
    )
    monkeypatch.setattr(
        bundle_module,
        "_build_htr_input_nontruncation_audit",
        forbidden,
    )
    monkeypatch.setattr(
        bundle_module,
        "build_embedding_cluster_feasibility_audit",
        forbidden,
    )
    monkeypatch.setattr(
        embedding_discovery,
        "MiniBatchKMeans",
        forbidden,
    )
    monkeypatch.setattr(
        embedding_discovery.np.linalg,
        "svd",
        forbidden,
    )
    monkeypatch.setattr(reusable, "_decode_state_tree", forbidden)
    monkeypatch.setattr(reusable, "_read_owner_parquet", forbidden)
    original_tree_validator = reusable._validate_registered_tree

    def stat_only_tree_validator(
        *,
        read_payload_bytes: bool,
        **kwargs: object,
    ):
        if read_payload_bytes:
            raise AssertionError(
                "unchanged reusable preflight reread registered payload bytes"
            )
        return original_tree_validator(
            read_payload_bytes=False,
            **kwargs,
        )

    monkeypatch.setattr(
        reusable,
        "_validate_registered_tree",
        stat_only_tree_validator,
    )
    original_hash = reusable._stable_file_sha256

    def no_bulk_hash(path: Path):
        supplied = Path(path)
        if (
            supplied.suffix in {".npy", ".parquet"}
            or supplied.name == "captured_state.json"
        ):
            raise AssertionError(
                "unchanged reusable preflight hashed bulk owner payload"
            )
        return original_hash(supplied)

    monkeypatch.setattr(reusable, "_stable_file_sha256", no_bulk_hash)
    reopen_started = time.perf_counter()
    reopened = reusable.load_reusable_preflight_acceptance(
        store_root=store,
        selector=selector,
        producer_identity=assembled_producer,
        owner_producer_identity=owner_producer,
        global_audit_producer_identity=global_producer,
    )
    reopen_elapsed = time.perf_counter() - reopen_started

    assert reopened.authentication_mode == (
        "prior_proof_stat_continuity"
    )
    assert reopened.global_audit_authentication_mode == (
        "prior_proof_stat_continuity"
    )
    assert reopened.preflight.authentication[
        "assembled_authentication_mode"
    ] == "prior_proof_stat_continuity"
    assert reopened.preflight.authentication["owner_fast_stat_count"] == 1
    assert reopened.preflight.authentication["owner_deep_auth_count"] == 0
    assert reopened.preflight.authentication[
        "owner_state_payloads_deserialized"
    ] == 0
    assert reopened.preflight.authentication[
        "bulk_owner_payload_read_during_unchanged_fast_path"
    ] is False
    assert reopened.preflight.identity() == first_identity
    assert reopened.state_bundle.content_sha256 == first_state_identity
    assert (
        reopened.prepared_context.content_root_sha256
        == first_context_identity
    )
    assert "captured_state" not in (
        reopened.preflight._owners[owner]._loaded
    )
    print(
        json.dumps(
            {
                "schema_version": (
                    "production_stage1_reusable_preflight_fixture_timing_v1"
                ),
                "cold_elapsed_seconds": cold_elapsed,
                "cold_payload_bytes": cold_payload_bytes,
                "reopen_elapsed_seconds": reopen_elapsed,
                "reopen_payload_bytes_read": (
                    reopened.payload_bytes_read
                    + reopened.global_audit_payload_bytes_read
                    + reopened.preflight.authentication[
                        "payload_bytes_read"
                    ]
                ),
                "owner_artifacts_reused": reopened.preflight.authentication[
                    "owner_fast_stat_count"
                ],
                "owner_artifacts_recomputed": reopened.preflight.authentication[
                    "owner_recomputed_count"
                ],
                "scientific_identity_unchanged": (
                    reopened.preflight.identity() == first_identity
                ),
                "bulk_payload_read_or_scientific_refit": False,
            },
            sort_keys=True,
        )
    )
