from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.config import AppliedInferenceConfig, ExperimentConfig
from oci.inference import production_stage1_legacy_scope_adapter as legacy_adapter
from oci.inference.embedding_native_proof_capture import (
    LOGICAL_FROZEN_EMBEDDING_CACHE_URI,
    canonical_logical_embedding_config,
)
from oci.inference.production_stage1_legacy_scope_adapter import (
    LegacyStage1RoleSpecificDeduplicationError,
    _load_scope_modeling_data,
    collect_and_merge_legacy_stage1_scope_attempts,
    publish_legacy_stage1_scope_descriptor,
    validate_legacy_stage1_scope_descriptor,
)
from oci.inference.production_stage1_bundle import load_applied_stage1_config
from oci.inference.production_stage1_config_wire import (
    production_stage1_effective_config_payload,
)
from oci.inference.production_stage1_scope_scheduler import (
    SpawnedStage1ScopeOrchestrator,
    Stage1ScopeAttemptStore,
    Stage1ScopeExecutionRequest,
    build_canonical_stage1_scope_plan,
)
from oci.inference.review_spent_evidence_provider import (
    SpentOnlyFrozenChunkEmbeddingCache,
)
from oci.inference.stage1_exact_inner_evidence import (
    CanonicalStage1SplitRegistry,
)
from oci.inference.stage1_upstream_gate_backend import (
    HistoricalStage1ConfigSnapshot,
)

_REQUEST_SHA = "b" * 64
_REGISTRY_SHA = "a" * 64


def _registry(*, outer_count: int = 2, inner_count: int = 4) -> dict:
    row_count = outer_count * 12
    rows = tuple(range(row_count))
    outer_heldout_by_fold: dict[int, tuple[int, ...]] = {}
    for outer_fold in range(1, outer_count + 1):
        start = (outer_fold - 1) * (row_count // outer_count)
        outer_heldout_by_fold[outer_fold] = tuple(range(start, start + row_count // outer_count))
    exact = CanonicalStage1SplitRegistry.build(
        dataset_row_ids=rows,
        outer_heldout_row_ids=outer_heldout_by_fold,
        inner_fold_count=inner_count,
        inner_seed_base=51_000,
    )
    outers = []
    for outer in exact.outer_splits:
        outers.append(
            {
                "outer_fold": outer.outer_fold,
                "fit_row_ids": list(outer.train_row_ids),
                "heldout_row_ids": list(outer.heldout_row_ids),
                "inner_folds": [
                    {
                        "inner_fold": inner.inner_fold,
                        "fit_row_ids": list(inner.fit_row_ids),
                        "heldout_row_ids": list(inner.heldout_row_ids),
                    }
                    for inner in outer.inner_splits
                ],
            }
        )
    return {
        "dataset_row_count": row_count,
        "inner_seed_base": 51_000,
        "outer_folds": outers,
    }


def _write_cache(
    root: Path,
    *,
    texts: list[str],
    config: AppliedInferenceConfig,
) -> SpentOnlyFrozenChunkEmbeddingCache:
    root.mkdir(parents=True)
    embeddings = np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4)
    np.save(root / "chunk_embeddings.npy", embeddings)
    np.save(root / "offsets.npy", np.arange(len(texts) + 1, dtype=np.int64))
    (root / "chunk_texts.jsonl").write_text(
        "".join(json.dumps({"chunks": [text]}) + "\n" for text in texts),
        encoding="utf-8",
    )
    embedding = config.architecture.multi_model_forest.embedding_contrast
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "num_samples": len(texts),
                "hidden_size": 4,
                "sentence_model_name": embedding.model_name,
                "chunk_size_words": embedding.chunk_size_words,
                "chunk_overlap_words": embedding.chunk_overlap_words,
                "max_chunks": embedding.max_chunks,
                "chunk_selection": embedding.chunk_selection,
                "normalize_embeddings": embedding.normalize_embeddings,
                "max_seq_length": embedding.max_seq_length,
            }
        ),
        encoding="utf-8",
    )
    return SpentOnlyFrozenChunkEmbeddingCache(root)


def _prepared(
    tmp_path: Path,
    *,
    label_flip_row: int | None = None,
    full_production_config: bool = False,
):
    registry = _registry(inner_count=5)
    if full_production_config:
        config = load_applied_stage1_config(
            Path(__file__).resolve().parents[1]
            / "example_configs"
            / "production_all_evidence_stage1_full.json"
        )
    else:
        config = AppliedInferenceConfig()
    config.cv_folds = 2
    config.architecture.multi_model_forest.candidate_consistency_inner_folds = 5
    plan = build_canonical_stage1_scope_plan(
        registry=registry,
        registry_content_sha256=_REGISTRY_SHA,
        global_seed=42,
        gpu_ids=(),
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=2,
        expected_inner_fold_count=5,
    )
    row_count = registry["dataset_row_count"]
    texts = [f"note-{row}" for row in range(row_count)]
    treatment = np.asarray([row % 2 for row in range(row_count)], dtype=float)
    outcome = np.asarray([(row // 2) % 2 for row in range(row_count)], dtype=float)
    if label_flip_row is not None:
        treatment[label_flip_row] = 1.0 - treatment[label_flip_row]
        outcome[label_flip_row] = 1.0 - outcome[label_flip_row]
    modeling = pd.DataFrame(
        {
            config.text_column: texts,
            config.treatment_column: treatment,
            config.outcome_column: outcome,
        }
    )
    cache = _write_cache(tmp_path / "cache", texts=texts, config=config)
    source_dataset = tmp_path / "label_cohort.parquet"
    effective = production_stage1_effective_config_payload(config)
    effective["dataset_path"] = str(source_dataset)
    audit_scopes = [
        {
            "scope_id": scope.scope_id,
            "cluster_fit_identity": {
                "scope_id": scope.scope_id,
                "content_sha256": hashlib.sha256(f"cluster:{scope.scope_id}".encode()).hexdigest(),
            },
        }
        for scope in plan.scopes
    ]
    return SimpleNamespace(
        request_sha256=_REQUEST_SHA,
        request={"effective_stage1_config": effective},
        registry=registry,
        registry_content_sha256=_REGISTRY_SHA,
        stage1_scope_plan=plan,
        modeling_data=modeling,
        config=config,
        embedding_cache_path=cache.cache_dir,
        embedding_cache=cache,
        embedding_cache_identity=cache.identity(),
        htr_model_path=tmp_path / "htr",
        htr_model_sha256="c" * 64,
        behavior_identity={"content_sha256": "d" * 64},
        embedding_cluster_feasibility_audit={
            "content_sha256": "e" * 64,
            "scopes": audit_scopes,
        },
        options=SimpleNamespace(
            num_workers=1,
            scope_workers_per_gpu=1,
            dataset_path=source_dataset,
        ),
    )


def test_full_production_request_and_legacy_descriptor_config_round_trip(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, full_production_config=True)
    effective = prepared.request["effective_stage1_config"]
    legacy_effective = effective["architecture"]["multi_model_agentic_forest"]
    integrated_effective = effective["architecture"]["multi_model_forest"]
    assert legacy_effective["fold_parallelism"] == "auto"
    assert "bow_fold_parallelism" not in legacy_effective
    assert "htr_fold_parallelism" not in legacy_effective
    assert integrated_effective["bow_fold_parallelism"] == "1"
    assert integrated_effective["htr_fold_parallelism"] == "1"
    parsed = ExperimentConfig.from_dict({"applied_inference": effective}).applied_inference
    assert asdict(parsed) == effective
    stage1_config = tmp_path / "stage1_config.json"
    stage1_config.write_text(
        json.dumps(
            {
                "model_type": "multi_model_forest",
                "stage": "stage1_bundle",
                "config": effective,
            }
        ),
        encoding="utf-8",
    )
    assert (
        asdict(HistoricalStage1ConfigSnapshot.from_path(stage1_config).applied_config())
        == effective
    )

    descriptor_set = publish_legacy_stage1_scope_descriptor(
        prepared=prepared,
        descriptor_root=(tmp_path / "descriptor").resolve(),
    )
    scope_id = "outer_001_full"
    descriptor = validate_legacy_stage1_scope_descriptor(
        descriptor_manifest_path=descriptor_set.descriptors[scope_id].manifest_path,
        expected_stage1_request_sha256=_REQUEST_SHA,
        expected_scope_id=scope_id,
    )
    request = _request_for_scope(
        descriptor,
        scope_id=scope_id,
        attempt_dir=tmp_path / "worker_attempt",
    )
    _modeling, private_config = _load_scope_modeling_data(
        descriptor=descriptor,
        request=request,
    )
    assert private_config.architecture.multi_model_forest.bow_fold_parallelism == "1"
    assert private_config.architecture.multi_model_forest.htr_fold_parallelism == "1"
    assert private_config.architecture.multi_model_agentic_forest.fold_parallelism == "auto"


def _request_for_scope(
    descriptor,
    *,
    scope_id: str,
    attempt_dir: Path,
) -> Stage1ScopeExecutionRequest:
    scope = descriptor.scope
    assignment = descriptor.assignment
    assert scope.scope_id == scope_id
    parameters = descriptor.worker_parameters()
    return Stage1ScopeExecutionRequest(
        attempt_dir=str(attempt_dir),
        plan_content_sha256=descriptor.plan_content_sha256,
        scope=scope.as_dict(),
        assignment=assignment.as_dict(),
        worker_target=f"{__name__}:_spawn_descriptor_probe_worker",
        worker_parameters=parameters,
        worker_parameters_sha256=hashlib.sha256(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        attempt_request_sha256="f" * 64,
    )


def _spawn_descriptor_probe_worker(
    request: Stage1ScopeExecutionRequest,
) -> dict:
    descriptor = validate_legacy_stage1_scope_descriptor(
        descriptor_manifest_path=request.worker_parameters["descriptor_manifest_path"],
        expected_stage1_request_sha256=request.worker_parameters["stage1_request_sha256"],
        expected_scope_id=request.scope_id,
    )
    modeling, config = _load_scope_modeling_data(descriptor=descriptor, request=request)
    fit = set(map(int, request.scope["fit_row_ids"]))
    heldout = set(map(int, request.scope["heldout_row_ids"]))
    finite = set(
        np.flatnonzero(
            modeling[[config.treatment_column, config.outcome_column]]
            .notna()
            .all(axis=1)
            .to_numpy()
        ).tolist()
    )
    visible = set(np.flatnonzero(modeling[config.text_column].map(bool).to_numpy()).tolist())
    expected_visible = fit if request.scope["scope_kind"] == "cumulative_spent" else fit | heldout
    if finite != fit or visible != expected_visible:
        raise RuntimeError("spawned probe observed a scope-isolation mismatch")
    root_files = {path.name for path in descriptor.root.rglob("*") if path.is_file()}
    if "label_cohort.parquet" in root_files:
        raise RuntimeError("source cohort entered child descriptor")
    return {
        "scope_id": request.scope_id,
        "finite_label_rows": sorted(finite),
        "visible_text_rows": sorted(visible),
        "heldout_labels_supplied": False,
    }


def _json_values_in_tree(root: Path):
    for path in sorted(root.rglob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def _nested_mappings(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _nested_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_mappings(child)


def test_private_descriptors_project_labels_text_preflight_and_cache(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    descriptor_set = publish_legacy_stage1_scope_descriptor(
        prepared=prepared,
        descriptor_root=(tmp_path / "recovery" / "descriptor").resolve(),
    )
    assert list(descriptor_set.descriptors) == [
        scope.scope_id for scope in prepared.stage1_scope_plan.physical_scopes
    ]
    assert descriptor_set.manifest["physical_scope_count"] == 14
    assert descriptor_set.manifest["logical_scope_count"] == 16
    assert "outer_001_hierarchy_epoch_001" not in descriptor_set.descriptors
    reused = next(
        row
        for row in descriptor_set.manifest["logical_physical_bindings"]
        if row["logical_scope_id"] == "outer_001_hierarchy_epoch_001"
    )
    assert reused == {
        "logical_scope_id": "outer_001_hierarchy_epoch_001",
        "physical_owner_scope_id": "outer_001_inner_005",
        "reuses_physical_fit": True,
    }
    source_path = str(prepared.options.dataset_path).encode()
    for scope_id in (
        "outer_001_full",
        "outer_001_inner_001",
        "outer_001_hierarchy_epoch_000",
    ):
        descriptor = descriptor_set.descriptors[scope_id]
        assert source_path not in b"".join(
            path.read_bytes() for path in descriptor.root.rglob("*") if path.is_file()
        )
        attempt_dir = tmp_path / "manual" / scope_id
        attempt_dir.mkdir(parents=True)
        request = _request_for_scope(descriptor, scope_id=scope_id, attempt_dir=attempt_dir)
        modeling, config = _load_scope_modeling_data(descriptor=descriptor, request=request)
        scope = descriptor.scope
        fit = set(scope.fit_row_ids)
        heldout = set(scope.heldout_row_ids)
        finite = set(
            np.flatnonzero(
                modeling[[config.treatment_column, config.outcome_column]]
                .notna()
                .all(axis=1)
                .to_numpy()
            )
        )
        visible = set(np.flatnonzero(modeling[config.text_column].map(bool).to_numpy()))
        assert finite == fit
        assert visible == (fit if scope.scope_kind == "cumulative_spent" else fit | heldout)
        projection = json.loads((descriptor.root / "cluster_preflight_projection.json").read_text())
        assert projection["scope_id"] == scope_id
        assert "scopes" not in projection
        cache = descriptor.manifest["embedding_cache"]
        offsets = np.load(
            descriptor.root / cache["relative_path"] / "offsets.npy",
            allow_pickle=False,
        )
        for row_id in set(range(len(modeling))) - fit:
            assert offsets[row_id] == offsets[row_id + 1]


def test_cluster_preflight_projection_uses_compact_canonical_json(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    descriptor_set = publish_legacy_stage1_scope_descriptor(
        prepared=prepared,
        descriptor_root=(tmp_path / "descriptor").resolve(),
    )
    projection_path = (
        descriptor_set.descriptors["outer_001_full"].root / "cluster_preflight_projection.json"
    )
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    expected = (
        json.dumps(
            projection,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    assert projection_path.read_bytes() == expected
    assert len(expected) < len(
        (json.dumps(projection, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    )
    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="unexpectedly large"):
        legacy_adapter._read_file_bytes(oversized)


def test_private_descriptor_serialization_and_tree_expose_one_scope_only(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    descriptor_set = publish_legacy_stage1_scope_descriptor(
        prepared=prepared,
        descriptor_root=(tmp_path / "descriptor").resolve(),
    )
    selected_scope_ids = (
        "outer_001_full",
        "outer_001_inner_001",
        "outer_001_hierarchy_epoch_000",
    )
    all_scope_ids = {scope.scope_id for scope in prepared.stage1_scope_plan.scopes}
    for scope_id in selected_scope_ids:
        descriptor = descriptor_set.descriptors[scope_id]
        scope = descriptor.scope
        peer_scope_ids = all_scope_ids - {scope_id}
        serialized_parameters = json.dumps(
            descriptor.worker_parameters(),
            sort_keys=True,
        )
        assert all(peer not in serialized_parameters for peer in peer_scope_ids)
        assert not (descriptor.root / "split_registry.json").exists()
        assert not (descriptor.root / "scope_plan.json").exists()
        assert (descriptor.root / "one_scope_authority.json").is_file()
        assert descriptor.authority["authorized_scope_count"] == 1
        assert descriptor.authority["other_scope_definitions_supplied"] is False
        assert descriptor.authority["other_scope_row_identities_supplied"] is False
        observed_scope_ids: set[str] = set()
        for value in _json_values_in_tree(descriptor.root):
            for mapping in _nested_mappings(value):
                if "scope_id" in mapping:
                    observed_scope_ids.add(str(mapping["scope_id"]))
                if "fit_row_ids" in mapping:
                    assert list(mapping["fit_row_ids"]) == list(scope.fit_row_ids)
                if "heldout_row_ids" in mapping:
                    assert list(mapping["heldout_row_ids"]) == list(scope.heldout_row_ids)
        assert observed_scope_ids == {scope_id}


def test_selected_legacy_scientific_path_does_not_access_global_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from oci.inference import production_stage1_bundle as bundle

    prepared = _prepared(tmp_path)
    descriptor_set = publish_legacy_stage1_scope_descriptor(
        prepared=prepared,
        descriptor_root=(tmp_path / "descriptor").resolve(),
    )

    class ReachedScientificModelBoundary(RuntimeError):
        pass

    def forbid_global_registry(*_args, **_kwargs):
        raise AssertionError("selected worker accessed the global split registry")

    def stop_before_scientific_fit(_path):
        raise ReachedScientificModelBoundary

    monkeypatch.setattr(bundle, "_registry_scopes", forbid_global_registry)
    monkeypatch.setattr(
        bundle,
        "_canonical_exact_registry_from_wrapper",
        forbid_global_registry,
    )
    monkeypatch.setattr(
        bundle,
        "PrivateHTRModelTreeSnapshot",
        stop_before_scientific_fit,
    )
    builder = bundle.ProductionStage1BundleBuilder.__new__(bundle.ProductionStage1BundleBuilder)
    for scope_id in (
        "outer_001_inner_001",
        "outer_001_hierarchy_epoch_000",
    ):
        descriptor = descriptor_set.descriptors[scope_id]
        selected = SimpleNamespace(
            selected_scope_authority=copy.deepcopy(dict(descriptor.authority)),
            selected_scope_spec=descriptor.scope,
            registry_content_sha256=prepared.registry_content_sha256,
            htr_model_path=prepared.htr_model_path,
        )
        with pytest.raises(ReachedScientificModelBoundary):
            builder._run_legacy_component(
                tmp_path / f"selected_{scope_id}",
                selected,
                selected_scope_id=scope_id,
            )


def test_hidden_parquet_column_and_heldout_label_drift_fail_closed(
    tmp_path: Path,
):
    first = _prepared(tmp_path / "first")
    scope = first.stage1_scope_plan.scope("outer_001_full")
    second = _prepared(tmp_path / "second", label_flip_row=scope.heldout_row_ids[0])
    first_set = publish_legacy_stage1_scope_descriptor(
        prepared=first,
        descriptor_root=(tmp_path / "first_descriptor").resolve(),
    )
    second_set = publish_legacy_stage1_scope_descriptor(
        prepared=second,
        descriptor_root=(tmp_path / "second_descriptor").resolve(),
    )
    first_registration = first_set.descriptors[scope.scope_id].manifest["files"]["fit_labels"]
    second_registration = second_set.descriptors[scope.scope_id].manifest["files"]["fit_labels"]
    assert first_registration["sha256"] == second_registration["sha256"]

    descriptor = first_set.descriptors[scope.scope_id]
    path = descriptor.root / first_registration["relative_path"]
    frame = pd.read_parquet(path)
    frame["true_ite_prob"] = 1.0
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="changed|hidden"):
        validate_legacy_stage1_scope_descriptor(
            descriptor_manifest_path=descriptor.manifest_path,
            expected_stage1_request_sha256=_REQUEST_SHA,
            expected_scope_id=scope.scope_id,
        )


def test_private_descriptor_rejects_extra_links_and_cache_paths_are_logical(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    descriptor_set = publish_legacy_stage1_scope_descriptor(
        prepared=prepared,
        descriptor_root=(tmp_path / "descriptor").resolve(),
    )
    scope_id = prepared.stage1_scope_plan.scopes[0].scope_id
    descriptor = descriptor_set.descriptors[scope_id]
    global_config = copy.deepcopy(
        prepared.config.architecture.multi_model_forest.embedding_contrast
    )
    private_config = copy.deepcopy(global_config)
    global_config.cache_dir = str(prepared.embedding_cache_path)
    private_config.cache_dir = str(descriptor.root / "private_embedding_cache")
    assert canonical_logical_embedding_config(global_config) == canonical_logical_embedding_config(
        private_config
    )
    assert (
        canonical_logical_embedding_config(private_config)["cache_dir"]
        == LOGICAL_FROZEN_EMBEDDING_CACHE_URI
    )

    (descriptor.root / "label_escape").symlink_to(prepared.options.dataset_path)
    with pytest.raises(ValueError, match="symbolic link|unregistered"):
        validate_legacy_stage1_scope_descriptor(
            descriptor_manifest_path=descriptor.manifest_path,
            expected_stage1_request_sha256=_REQUEST_SHA,
            expected_scope_id=scope_id,
        )


def test_spawned_probe_and_sibling_attempt_resume_use_stable_recovery(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    recovery = tmp_path / "recovery"
    descriptor_set = publish_legacy_stage1_scope_descriptor(
        prepared=prepared,
        descriptor_root=(recovery / "descriptor").resolve(),
    )
    parameters = descriptor_set.worker_parameters_by_scope()
    target = f"{__name__}:_spawn_descriptor_probe_worker"
    first = SpawnedStage1ScopeOrchestrator(
        plan=prepared.stage1_scope_plan,
        attempt_root=recovery / "attempts",
        progress_path=tmp_path / "modeling_attempt_1" / "progress.json",
        worker_target=target,
        worker_parameters_by_scope=parameters,
    ).run()
    attempt_directories = {attempt.attempt_dir for attempt in first}
    second = SpawnedStage1ScopeOrchestrator(
        plan=prepared.stage1_scope_plan,
        attempt_root=recovery / "attempts",
        progress_path=tmp_path / "modeling_attempt_2" / "progress.json",
        worker_target=target,
        worker_parameters_by_scope=parameters,
    ).run()
    assert {attempt.attempt_dir for attempt in second} == attempt_directories
    assert len(tuple((recovery / "attempts").glob("*/attempt_*"))) == len(
        prepared.stage1_scope_plan.physical_scopes
    )
    bindings = Stage1ScopeAttemptStore(
        recovery / "attempts", prepared.stage1_scope_plan
    ).validate_logical_bindings()
    assert bindings.manifest["logical_scope_count"] == len(
        prepared.stage1_scope_plan.scopes
    )
    assert bindings.manifest["physical_fit_count"] == len(
        prepared.stage1_scope_plan.physical_scopes
    )
    merge_root = (tmp_path / "must_not_publish_role_specific_merge").resolve()
    with pytest.raises(
        LegacyStage1RoleSpecificDeduplicationError,
        match="role-neutral-binding-set|role_neutral_binding_set|ten fit-side",
    ):
        collect_and_merge_legacy_stage1_scope_attempts(
            prepared=prepared,
            attempts=first,
            merge_root=merge_root,
        )
    assert not merge_root.exists()
    first_scope = prepared.stage1_scope_plan.scopes[0].scope_id
    assert (
        Stage1ScopeAttemptStore(recovery / "attempts", prepared.stage1_scope_plan).reusable_attempt(
            scope_id=first_scope,
            worker_target=target,
            worker_parameters=parameters[first_scope],
        )
        is not None
    )


def test_descriptor_publication_preserves_and_reuses_completed_scopes_after_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared = _prepared(tmp_path)
    descriptor_root = (tmp_path / "recovery" / "descriptor").resolve()
    original_writer = legacy_adapter._write_scope_descriptor
    writes = 0

    def interrupt_after_three(**kwargs):
        nonlocal writes
        if writes == 3:
            raise KeyboardInterrupt("fixture interruption")
        writes += 1
        return original_writer(**kwargs)

    monkeypatch.setattr(legacy_adapter, "_write_scope_descriptor", interrupt_after_three)
    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        publish_legacy_stage1_scope_descriptor(
            prepared=prepared,
            descriptor_root=descriptor_root,
        )

    assert not (descriptor_root / "descriptor_set_manifest.json").exists()
    completed_scope_ids = [
        scope.scope_id
        for scope in prepared.stage1_scope_plan.physical_scopes[:3]
    ]
    preserved: dict[str, tuple[str, int]] = {}
    for scope_id in completed_scope_ids:
        scope_root = descriptor_root / scope_id
        for path in sorted(item for item in scope_root.rglob("*") if item.is_file()):
            relative = path.relative_to(descriptor_root).as_posix()
            preserved[relative] = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                int(path.stat().st_ino),
            )

    recovery_root = descriptor_root.parent / (f".{descriptor_root.name}.scope_descriptor_attempts")
    interrupted_scope = prepared.stage1_scope_plan.physical_scopes[3].scope_id
    partial_attempts = tuple((recovery_root / interrupted_scope).glob("attempt_*"))
    assert len(partial_attempts) == 1
    assert not (
        partial_attempts[0] / legacy_adapter.LEGACY_STAGE1_SCOPE_DESCRIPTOR_MANIFEST
    ).exists()

    monkeypatch.setattr(legacy_adapter, "_write_scope_descriptor", original_writer)
    descriptor_set = publish_legacy_stage1_scope_descriptor(
        prepared=prepared,
        descriptor_root=descriptor_root,
    )

    assert list(descriptor_set.descriptors) == [
        scope.scope_id for scope in prepared.stage1_scope_plan.physical_scopes
    ]
    for relative, (digest, inode) in preserved.items():
        path = descriptor_root / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert int(path.stat().st_ino) == inode
    assert partial_attempts[0].is_dir()
    assert not any(partial_attempts[0].iterdir())
