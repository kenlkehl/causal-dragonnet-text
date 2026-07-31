from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

import oci.inference.production_stage1_cluster_preflight_artifact_v2 as v2
import oci.inference.production_stage1_scope_scheduler as scope_scheduler
import oci.inference.role_neutral_embedding_group_execution as embedding_execution
from oci.inference.role_neutral_embedding_group_execution import (
    load_canonical_clustered_preflight_state_bundle,
    load_canonical_clustered_preflight_state_bundle_for_scientific_migration,
    seal_canonical_clustered_preflight_state_bundle,
    seal_canonical_clustered_preflight_scope_state,
)
from oci.inference.production_stage1_scope_scheduler import (
    Stage1PhysicalFitIdentity,
    Stage1ScopePlan,
)
from oci.inference.stage1_exact_inner_evidence import (
    row_order_fingerprint,
)
from tests.test_production_stage1_cluster_preflight_artifact import (
    _fixture_values as _v1_fixture_values,
)
from tests.test_role_neutral_embedding_group_execution import (
    _one_physical_group_plan,
    _preflight_and_states,
    _request as _embedding_request,
    _texts,
    _write_cache,
)


def _addressed(body):
    return {**body, "content_sha256": v2._sha256_json(body)}


def _rehash_addressed(value):
    body = {key: child for key, child in value.items() if key != "content_sha256"}
    value["content_sha256"] = v2._sha256_json(body)


def _rehash_request(value):
    body = {key: child for key, child in value.items() if key != "request_sha256"}
    value["request_sha256"] = v2._sha256_json(body)


def _fit_identity(owner: str, index: int):
    raw = [
        {
            "contrast_family": "cluster_local_treatment_contrast_basis",
            "name": f"raw-{index}-0",
            "scores": [index, 0],
        },
        {
            "contrast_family": ("cluster_local_residualized_interaction_contrast_basis"),
            "name": f"raw-{index}-1",
            "scores": [index, 1],
        },
    ]
    semantic = [
        {
            "contrast_family": row["contrast_family"],
            "name": row["name"],
            "concept_probe_scores": [{"phrase": f"owner {index} concept {position}", "score": 1.0}],
        }
        for position, row in enumerate(raw)
    ]
    final = {
        "embedding_clustered": [{"atom_kind": "embedding_contrast", "content": semantic[0]}],
        "tfidf_semantic_retrieval": [
            {
                "atom_kind": "tfidf_semantic_retrieval_contrast",
                "content": semantic[1],
            }
        ],
    }
    body = {
        "schema_version": "fixture_fit_identity_v2",
        "scope_id": owner,
        "fit_row_ids": [index, index + 100],
        "fit_row_order_fingerprint": v2._sha256_json([index, index + 100]),
        "canonical_group_seed": 42000 + index,
        "ordered_fit_row_seed_policy": "canonical_ordered_fit_rows_group_seed_v1",
        "kmeans": {"cluster_centers": {"sha256": f"{index:064x}"[-64:]}},
        "svd_families": [{"family_key": "treatment"}],
        "raw_cluster_concepts": raw,
        "raw_cluster_concepts_sha256": v2._sha256_json(raw),
        "semantic_cluster_concepts": semantic,
        "semantic_cluster_concepts_sha256": v2._sha256_json(semantic),
        "final_catalog_concepts": final,
        "final_catalog_concepts_sha256": v2._sha256_json(final),
    }
    return _addressed(body)


def _portable_fixture():
    scope_order = [f"scope_{index:03d}" for index in range(40)]
    physical_order = scope_order[:35]
    identities = {owner: _fit_identity(owner, index) for index, owner in enumerate(physical_order)}
    scopes = []
    for index, scope_id in enumerate(scope_order):
        owner = scope_id if index < 35 else physical_order[index - 35]
        binding_body = {
            "schema_version": ("production_stage1_cluster_preflight_physical_binding_v2"),
            "logical_scope_id": scope_id,
            "physical_owner_scope_id": owner,
            "reuses_physical_fit": scope_id != owner,
        }
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
                "context_epoch": None if index < 30 else index % 2,
                "provider_inner_fold": None if index < 30 else index % 5,
                "fit_row_count": 2,
                "fit_row_order_fingerprint": identities[owner]["fit_row_order_fingerprint"],
                "heldout_row_count": 1,
                "heldout_row_order_fingerprint": v2._sha256_json(["heldout", index]),
                "canonical_group_seed": identities[owner]["canonical_group_seed"],
                "physical_fit_binding": _addressed(binding_body),
                "cluster_fit_identity": copy.deepcopy(identities[owner]),
                "token_bounded_row_count": 0,
                "uncapped_semantic_projection": True,
            }
        )
    audit_body = {
        "schema_version": "fixture_cluster_audit_v2",
        "embedding_cache_identity_sha256": "a" * 64,
        "scope_count": len(scopes),
        "scope_order": scope_order,
        "scopes": scopes,
        "physical_fit_count": len(physical_order),
        "deduplicated_fit_count": len(scopes) - len(physical_order),
        "physical_scope_order": physical_order,
        "all_required_scopes_passed": True,
    }
    audit = _addressed(audit_body)
    reference = v2.build_portable_cluster_preflight_reference(audit)
    _old_audit, request = _v1_fixture_values()
    request["embedding_cluster_feasibility_audit"] = reference
    request_body = {key: child for key, child in request.items() if key != "request_sha256"}
    request["request_sha256"] = v2._sha256_json(request_body)
    return audit, request


@pytest.fixture
def portable_validators(monkeypatch):
    monkeypatch.setattr(
        v2,
        "_validate_scientific_audit",
        lambda audit, **_kwargs: copy.deepcopy(dict(audit)),
    )

    def validate_request(request, *, expected_reference):
        assert request["embedding_cluster_feasibility_audit"] == expected_reference
        return copy.deepcopy(dict(request))

    monkeypatch.setattr(
        v2,
        "_validate_stage1_request_with_reference",
        validate_request,
    )


def _seal(tmp_path: Path, *, parquet_compression: str = "zstd"):
    audit, request = _portable_fixture()
    artifact = v2.seal_portable_production_stage1_cluster_preflight_artifact(
        output_dir=(tmp_path / "portable_preflight").resolve(),
        audit=audit,
        stage1_request=request,
        config={"fixture": True},
        registry={"fixture": True},
        registry_content_sha256="b" * 64,
        embedding_cache_identity={"fixture": True},
        parquet_compression=parquet_compression,
    )
    return audit, request, artifact


def _load(path: Path, request):
    return v2.load_portable_production_stage1_cluster_preflight_artifact(
        manifest_path=path,
        config={"fixture": True},
        registry={"fixture": True},
        registry_content_sha256="b" * 64,
        embedding_cache_identity={"fixture": True},
        expected_stage1_request=request,
    )


def _make_writable(root: Path):
    root.chmod(stat.S_IRWXU)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(stat.S_IRWXU)
        else:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _make_read_only(root: Path):
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.chmod(v2._READ_ONLY_DIRECTORY_MODE)
        elif not path.is_symlink():
            path.chmod(v2._READ_ONLY_FILE_MODE)
    root.chmod(v2._READ_ONLY_DIRECTORY_MODE)


def test_portable_preflight_is_compact_deduplicated_and_lossless(
    tmp_path: Path,
    portable_validators,
    monkeypatch,
):
    audit, request, artifact = _seal(tmp_path)
    try:
        index = dict(artifact.audit)
        assert index["logical_scope_count"] == 40
        assert index["physical_fit_count"] == 35
        assert index["deduplicated_logical_scope_count"] == 5
        assert len(index["logical_scopes"]) == 40
        assert len(index["physical_fits"]) == 35
        assert not any(
            field in physical["compact_fit_identity"]
            for physical in index["physical_fits"]
            for field in v2._CONCEPT_FIELDS
        )
        payloads = sorted(
            (artifact.root / v2.PORTABLE_CLUSTER_PREFLIGHT_CONCEPT_DIRECTORY).glob("*.parquet")
        )
        assert len(payloads) == 35
        assert not (artifact.root / "cluster_feasibility_audit.json").exists()
        request_bytes = artifact.stage1_request_path.read_text(encoding="utf-8")
        assert "raw_cluster_concepts" not in request_bytes
        assert "semantic_cluster_concepts" not in request_bytes
        assert "final_catalog_concepts" not in request_bytes
        assert request["embedding_cluster_feasibility_audit"] == artifact.reference

        reads = 0
        original_reader = v2._read_owner_parquet

        def counted_reader(*args, **kwargs):
            nonlocal reads
            reads += 1
            return original_reader(*args, **kwargs)

        monkeypatch.setattr(v2, "_read_owner_parquet", counted_reader)
        reopened = _load(artifact.manifest_path, request)
        assert reads == 0
        owner = "scope_000"
        alias = "scope_035"
        expected_owner = audit["scopes"][0]["cluster_fit_identity"]
        assert reopened.owner_fit_identity(owner) == expected_owner
        assert reads == 1
        assert reopened.logical_scope_record(alias)["cluster_fit_identity"] == expected_owner
        assert reads == 1
        assert reopened.logical_scope_record(owner) == audit["scopes"][0]
    finally:
        _make_writable(artifact.root)


def test_portable_preflight_relocates_without_changing_scientific_identity(
    tmp_path: Path,
    portable_validators,
):
    _audit, request, artifact = _seal(tmp_path)
    relocated_request = copy.deepcopy(request)
    relocated_request["dataset"]["path"] = "/another/cohort.parquet"
    relocated_request["source_config"]["path"] = "/another/profile.json"
    relocated_request["embedding_cache"]["path"] = "/another/cache"
    relocated_request["embedding_cache"]["production_cache_build_identity"][
        "cache_path"
    ] = "/another/cache"
    relocated_request["htr_model"]["path"] = "/another/htr"
    relocated_request["runtime"]["preflight_workers"] = 17
    relocated_request["effective_stage1_config"].update(
        {
            "dataset_path": "/another/cohort.parquet",
            "cache_dir": "/another/cache",
            "device": "cuda:7",
            "workers": 17,
        }
    )
    relocation = relocated_request["embedding_cache"]["authenticated_relocation"]
    relocation["root"] = "/another/relocation"
    relocation["cache_dir"] = "/another/relocation/cache"
    relocation["prepared_cohort_path"] = "/another/relocation/cohort.parquet"
    relocation["attestation_path"] = "/another/relocation/attestation.json"
    relocation["terminal_manifest_path"] = "/another/relocation/manifest.json"
    relocation["source_cache_identity_sha256"] = "0" * 64
    relocation["attestation_sha256"] = "1" * 64
    relocation["terminal_manifest_sha256"] = "2" * 64
    relocation["cache_build_identity"]["cache_path"] = "/another/cache"
    relocated_request["stage1_scope_plan"]["gpu_ids"] = [7]
    relocated_request["stage1_scope_plan"]["scope_workers_per_gpu"] = 3
    relocated_request["stage1_scope_plan"]["assignments"].reverse()
    _rehash_addressed(relocated_request["stage1_scope_plan"])
    _rehash_request(relocated_request)
    relocated = tmp_path / "relocated" / artifact.root.name
    relocated.parent.mkdir()
    shutil.copytree(artifact.root, relocated, copy_function=shutil.copy2)
    try:
        reopened = _load(
            relocated / v2.PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME,
            relocated_request,
        )
        before = artifact.identity()
        after = reopened.identity()
        assert (
            before["path_neutral_scientific_content_sha256"]
            == after["path_neutral_scientific_content_sha256"]
        )
        assert (
            before["portable_audit_reference_content_sha256"]
            == after["portable_audit_reference_content_sha256"]
        )
        assert before["root"] != after["root"]
    finally:
        _make_writable(artifact.root)
        _make_writable(relocated)


def test_physical_parquet_compression_is_explicit_and_science_neutral(
    tmp_path: Path,
    portable_validators,
):
    seal_parameter = inspect.signature(
        v2.seal_portable_production_stage1_cluster_preflight_artifact
    ).parameters["parquet_compression"]
    assert seal_parameter.default is inspect.Parameter.empty
    assert seal_parameter.kind is inspect.Parameter.KEYWORD_ONLY

    roots = {
        compression: tmp_path / compression
        for compression in sorted(v2.SUPPORTED_PORTABLE_CLUSTER_PREFLIGHT_PARQUET_COMPRESSIONS)
    }
    for root in roots.values():
        root.mkdir()
    sealed = {
        compression: _seal(root, parquet_compression=compression)[2]
        for compression, root in roots.items()
    }
    try:
        identities = {compression: artifact.identity() for compression, artifact in sealed.items()}
        manifests = {
            compression: json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
            for compression, artifact in sealed.items()
        }
        assert (
            len(
                {
                    identity["path_neutral_scientific_content_sha256"]
                    for identity in identities.values()
                }
            )
            == 1
        )
        assert (
            len(
                {
                    identity["portable_audit_reference_content_sha256"]
                    for identity in identities.values()
                }
            )
            == 1
        )
        assert (
            len(
                {identity["compact_audit_index_content_sha256"] for identity in identities.values()}
            )
            == 1
        )
        assert len(
            {identity["payload_inventory_content_sha256"] for identity in identities.values()}
        ) == len(sealed)
        assert {
            compression: manifest["physical_storage"]["parquet_compression"]
            for compression, manifest in manifests.items()
        } == {compression: compression for compression in sealed}
        physical_byte_inventories = {
            compression: [
                (row["size_bytes"], row["sha256"])
                for row in manifest["files"]
                if row["kind"] == "physical_owner_concepts"
            ]
            for compression, manifest in manifests.items()
        }
        assert physical_byte_inventories["none"] != physical_byte_inventories["zstd"]
        owner = str(next(iter(sealed.values())).audit["physical_scope_order"][0])
        assert sealed["none"].owner_fit_identity(owner) == sealed["zstd"].owner_fit_identity(owner)
    finally:
        for artifact in sealed.values():
            _make_writable(artifact.root)


def test_portable_preflight_rejects_unconfigured_physical_compression(
    tmp_path: Path,
    portable_validators,
):
    audit, request = _portable_fixture()
    output = (tmp_path / "unsupported_compression").resolve()
    with pytest.raises(ValueError, match="explicit supported value"):
        v2.seal_portable_production_stage1_cluster_preflight_artifact(
            output_dir=output,
            audit=audit,
            stage1_request=request,
            config={"fixture": True},
            registry={"fixture": True},
            registry_content_sha256="b" * 64,
            embedding_cache_identity={"fixture": True},
            parquet_compression="gzip",
        )
    assert not output.exists()


@pytest.mark.parametrize("parquet_compression", ("none", "zstd"))
def test_every_owner_payload_byte_is_authenticated_for_each_storage_mode(
    tmp_path: Path,
    portable_validators,
    parquet_compression: str,
):
    _audit, request, artifact = _seal(
        tmp_path,
        parquet_compression=parquet_compression,
    )
    _make_writable(artifact.root)
    payload = next(
        (artifact.root / v2.PORTABLE_CLUSTER_PREFLIGHT_CONCEPT_DIRECTORY).glob("*.parquet")
    )
    original = bytearray(payload.read_bytes())
    original[len(original) // 2] ^= 1
    payload.write_bytes(original)
    _make_read_only(artifact.root)
    try:
        with pytest.raises(ValueError, match="registered bytes changed"):
            _load(artifact.manifest_path, request)
    finally:
        _make_writable(artifact.root)


def test_physical_storage_attestation_must_match_parquet_bytes(
    tmp_path: Path,
    portable_validators,
):
    _audit, request, artifact = _seal(tmp_path, parquet_compression="zstd")
    _make_writable(artifact.root)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    manifest["physical_storage"]["parquet_compression"] = "none"
    manifest_body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    manifest["content_sha256"] = v2._sha256_json(manifest_body)
    artifact.manifest_path.write_text(
        v2._canonical_json(manifest) + "\n",
        encoding="utf-8",
    )
    _make_read_only(artifact.root)
    try:
        with pytest.raises(ValueError, match="physical compression differs"):
            _load(artifact.manifest_path, request)
    finally:
        _make_writable(artifact.root)


def test_role_neutral_state_sealing_consumes_lazy_portable_owner_handle(
    tmp_path: Path,
    portable_validators,
    monkeypatch,
):
    plan = _one_physical_group_plan()
    embedding_request = _embedding_request(plan)
    texts = _texts()
    cache = _write_cache(tmp_path / "cache", texts=texts)
    legacy, _legacy_state, kmeans, svds = _preflight_and_states(
        tmp_path=tmp_path / "legacy",
        request=embedding_request,
        cache=cache,
    )
    source_audit = copy.deepcopy(dict(legacy.audit))
    scope = source_audit["scopes"][0]
    fit = scope["cluster_fit_identity"]
    treatment_family = "cluster_local_treatment_contrast_basis"
    fit["raw_cluster_concepts"] = [{"contrast_family": treatment_family, "complete": True}]
    fit["raw_cluster_concepts_sha256"] = v2._sha256_json(fit["raw_cluster_concepts"])
    fit["semantic_cluster_concepts"] = [{"contrast_family": treatment_family, "complete": True}]
    fit["semantic_cluster_concepts_sha256"] = v2._sha256_json(fit["semantic_cluster_concepts"])
    fit_body = {key: child for key, child in fit.items() if key != "content_sha256"}
    fit["content_sha256"] = v2._sha256_json(fit_body)
    owner = embedding_request.physical_owner.scope_id
    complete_scopes = []
    for member in plan.scopes:
        member_scope = copy.deepcopy(scope)
        member_scope.update(
            {
                "scope_id": member.scope_id,
                "scope_kind": member.scope_kind,
                "outer_fold": member.outer_fold,
                "inner_fold": member.inner_fold,
                "context_epoch": member.context_epoch,
                "provider_inner_fold": member.provider_inner_fold,
                "fit_row_count": member.fit_row_count,
                "fit_row_order_fingerprint": row_order_fingerprint(
                    member.fit_row_ids
                ),
                "canonical_group_seed": member.scope_seed,
                "heldout_row_count": len(
                    member.heldout_row_ids
                ),
                "heldout_row_order_fingerprint": row_order_fingerprint(
                    member.heldout_row_ids
                ),
            }
        )
        binding_body = {
            "schema_version": (
                "production_stage1_cluster_preflight_physical_binding_v2"
            ),
            "logical_scope_id": member.scope_id,
            "physical_owner_scope_id": owner,
            "reuses_physical_fit": member.scope_id != owner,
        }
        member_scope["physical_fit_binding"] = _addressed(
            binding_body
        )
        complete_scopes.append(member_scope)
    source_audit.update(
        {
            "scope_order": [
                member.scope_id for member in plan.scopes
            ],
            "scopes": complete_scopes,
            "physical_fit_count": 1,
            "deduplicated_fit_count": len(plan.scopes) - 1,
            "physical_scope_order": [owner],
        }
    )
    audit_body = {key: child for key, child in source_audit.items() if key != "content_sha256"}
    source_audit["content_sha256"] = v2._sha256_json(audit_body)
    reference = v2.build_portable_cluster_preflight_reference(source_audit)
    _unused, stage1_request = _v1_fixture_values(scope_count=1)
    stage1_request["embedding_cluster_feasibility_audit"] = reference
    stage1_request["stage1_scope_plan"] = plan.as_dict()
    request_body = {key: child for key, child in stage1_request.items() if key != "request_sha256"}
    stage1_request["request_sha256"] = v2._sha256_json(request_body)
    portable = v2.seal_portable_production_stage1_cluster_preflight_artifact(
        output_dir=(tmp_path / "portable_state_preflight").resolve(),
        audit=source_audit,
        stage1_request=stage1_request,
        config={"fixture": True},
        registry={"fixture": True},
        registry_content_sha256="b" * 64,
        embedding_cache_identity={"fixture": True},
        parquet_compression="zstd",
    )
    try:
        state = seal_canonical_clustered_preflight_scope_state(
            output_root=(tmp_path / "portable_cluster_state").resolve(),
            preflight=portable,
            request=embedding_request,
            kmeans_state=kmeans,
            svd_states=svds,
        )
        assert state.scope_record["cluster_fit_identity"] == fit
        assert portable.owner_fit_identity(owner) == fit
        captured = {
            owner: {
                "schema_version": (
                    "production_stage1_cluster_preflight_scope_state_capture_v2"
                ),
                "scope_id": owner,
                "cluster_fit_identity_content_sha256": fit["content_sha256"],
                "kmeans_state": kmeans,
                "svd_states": svds,
                "captured_from_canonical_preflight_fit": True,
                "refit_performed_for_state_capture": False,
            }
        }
        bundle = seal_canonical_clustered_preflight_state_bundle(
            output_root=(tmp_path / "portable_state_bundle").resolve(),
            preflight=portable,
            plan=plan,
            captured_scope_states=captured,
        )
        portable._owner_fit_cache.clear()
        reads = 0
        original_reader = v2._read_owner_parquet

        def counted_reader(*args, **kwargs):
            nonlocal reads
            reads += 1
            return original_reader(*args, **kwargs)

        monkeypatch.setattr(v2, "_read_owner_parquet", counted_reader)
        reopened = load_canonical_clustered_preflight_state_bundle(
            manifest_path=(
                bundle.root / "cluster_state_bundle_manifest.json"
            ),
            preflight=portable,
            plan=plan,
        )
        assert reads == 0
        assert set(reopened.states) == {owner}
        assert reopened.manifest_path_for_owner(owner).is_file()
        assert reads == 0
        loaded = reopened.load_state_for_owner(owner)
        assert loaded.content_sha256 == state.content_sha256
        assert reads == 1
        assert reopened.load_state_for_owner(owner) is loaded
        assert reads == 1

        changed_fit_identity = Stage1PhysicalFitIdentity(
            architecture_identity="4" * 64,
            target=plan.physical_fit_identity.target,
            scientific_configuration_identity="5" * 64,
            producer_identity="6" * 64,
            runtime_compatibility_class=(
                plan.physical_fit_identity.runtime_compatibility_class
            ),
        )
        changed_body = scope_scheduler._stage1_scope_plan_body(
            registry_content_sha256=plan.registry_content_sha256,
            global_seed=plan.global_seed,
            review_rounds=plan.review_rounds,
            initial_training_partitions=(
                plan.initial_training_partitions
            ),
            physical_fit_identity=changed_fit_identity,
            gpu_ids=(7,),
            scope_workers_per_gpu=3,
            scopes=plan.scopes,
            assignments=plan.assignments,
        )
        changed_plan = Stage1ScopePlan(
            registry_content_sha256=plan.registry_content_sha256,
            global_seed=plan.global_seed,
            review_rounds=plan.review_rounds,
            initial_training_partitions=(
                plan.initial_training_partitions
            ),
            physical_fit_identity=changed_fit_identity,
            gpu_ids=(7,),
            scope_workers_per_gpu=3,
            scopes=plan.scopes,
            assignments=plan.assignments,
            content_sha256=v2._sha256_json(changed_body),
        )
        changed_plan.as_dict()

        original_np_load = embedding_execution.np.load

        def no_path_or_mmap_load(source, *args, **kwargs):
            assert not isinstance(source, (str, os.PathLike))
            assert "mmap_mode" not in kwargs
            return original_np_load(source, *args, **kwargs)

        monkeypatch.setattr(
            embedding_execution.np,
            "load",
            no_path_or_mmap_load,
        )
        migrated = (
            load_canonical_clustered_preflight_state_bundle_for_scientific_migration(
                manifest_path=(
                    bundle.root
                    / "cluster_state_bundle_manifest.json"
                ),
                preflight=portable,
                current_plan=changed_plan,
                expected_source_plan_scientific_content_sha256=(
                    plan.scientific_content_sha256
                ),
            )
        )
        assert set(migrated.states) == {owner}
        assert (
            migrated.load_state_for_owner(owner).content_sha256
            == state.content_sha256
        )
    finally:
        _make_writable(portable.root)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "extra",
        "tamper",
        "substitute",
        "reordered",
        "symlink",
        "hardlink",
    ),
)
def test_portable_preflight_fails_closed_for_tree_or_payload_mutation(
    tmp_path: Path,
    portable_validators,
    mutation: str,
):
    _audit, request, artifact = _seal(tmp_path)
    _make_writable(artifact.root)
    payloads = sorted(
        (artifact.root / v2.PORTABLE_CLUSTER_PREFLIGHT_CONCEPT_DIRECTORY).glob("*.parquet")
    )
    if mutation == "missing":
        payloads[0].unlink()
    elif mutation == "extra":
        (artifact.root / "extra.bin").write_bytes(b"extra")
    elif mutation == "tamper":
        payloads[0].write_bytes(payloads[0].read_bytes() + b"x")
    elif mutation == "substitute":
        payloads[0].write_bytes(payloads[1].read_bytes())
    elif mutation == "reordered":
        index = json.loads(artifact.audit_path.read_text(encoding="utf-8"))
        index["logical_scopes"].reverse()
        artifact.audit_path.write_text(
            json.dumps(index, sort_keys=True),
            encoding="utf-8",
        )
    elif mutation == "symlink":
        payloads[0].unlink()
        payloads[0].symlink_to(payloads[1].name)
    else:
        payloads[0].unlink()
        os.link(payloads[1], payloads[0])
    _make_read_only(artifact.root)
    with pytest.raises((FileNotFoundError, ValueError)):
        _load(artifact.manifest_path, request)
    _make_writable(artifact.root)


@pytest.mark.parametrize(
    "target_name",
    (
        v2.PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME,
        v2.PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_NAME,
        v2.PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_NAME,
    ),
)
def test_portable_preflight_rejects_json_changed_after_byte_authentication(
    tmp_path: Path,
    portable_validators,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    _audit, request, artifact = _seal(tmp_path)
    original_reader = v2._read_json
    mutated = False

    def mutate_after_read(path: Path, *, label: str):
        nonlocal mutated
        value = original_reader(path, label=label)
        if not mutated and path.name == target_name:
            mutated = True
            path.chmod(0o600)
            with path.open("ab") as handle:
                handle.write(b" ")
                handle.flush()
                os.fsync(handle.fileno())
            path.chmod(v2._READ_ONLY_FILE_MODE)
        return value

    monkeypatch.setattr(v2, "_read_json", mutate_after_read)
    try:
        with pytest.raises(ValueError, match="changed after authentication"):
            _load(artifact.manifest_path, request)
        assert mutated is True
    finally:
        _make_writable(artifact.root)


def test_lazy_owner_handle_rejects_payload_changed_after_fresh_load(
    tmp_path: Path,
    portable_validators,
) -> None:
    _audit, request, artifact = _seal(tmp_path)
    reopened = _load(artifact.manifest_path, request)
    owner = str(reopened.audit["physical_scope_order"][0])
    record = next(
        row for row in reopened.audit["physical_fits"] if row["physical_owner_scope_id"] == owner
    )
    payload = reopened.root / str(record["payload_relative_path"])
    payload.chmod(0o600)
    with payload.open("ab") as handle:
        handle.write(b" ")
        handle.flush()
        os.fsync(handle.fileno())
    payload.chmod(v2._READ_ONLY_FILE_MODE)
    try:
        with pytest.raises(ValueError, match="changed after authentication"):
            reopened.owner_fit_identity(owner)
    finally:
        _make_writable(artifact.root)
