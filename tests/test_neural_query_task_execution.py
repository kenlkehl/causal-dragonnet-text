from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oci.config import TfidfNuisanceStackScientificConfig
from oci.inference import neural_query_context_backend as context_module
from oci.inference import neural_query_task_execution as task_execution_module
from oci.inference.neural_query_agentic_forest import (
    NeuralQueryAgenticForestConfig,
)
from oci.inference.neural_query_context_backend import (
    ContextFitNeuralQueryService,
)
from oci.inference.neural_query_operational_controls import (
    ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA,
    RoleNeutralNeuralQueryOperationalControls,
)
from oci.inference.neural_query_task_execution import (
    NeuralQueryAuthenticatedCacheReference,
    execute_bounded_neural_query_tasks,
)
from oci.inference.review_spent_evidence_provider import (
    SpentOnlyFrozenChunkEmbeddingCache,
)


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _controls(**changes) -> RoleNeutralNeuralQueryOperationalControls:
    values = {
        "inner_fold_parallelism": 4,
        "fold_parallel_backend": "processes",
        "fold_slots_per_device": 2,
        "bank_parallelism": 2,
        "worker_cpu_threads": 1,
        "schema_version": (
            ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA
        ),
    }
    values.update(changes)
    return RoleNeutralNeuralQueryOperationalControls.from_mapping(values)


def _barrier_echo(task, device):
    root = Path(task["barrier_root"])
    (root / f"{int(task['index']):03d}.ready").write_text(
        "ready\n",
        encoding="utf-8",
    )
    deadline = time.monotonic() + 30.0
    while len(tuple(root.glob("*.ready"))) < int(task["expected"]):
        if time.monotonic() >= deadline:
            raise RuntimeError("test neural-query process barrier timed out")
        time.sleep(0.01)
    time.sleep(float(task["delay"]))
    return {
        "index": int(task["index"]),
        "device_seen": str(device),
    }


def test_controls_are_closed_and_reject_cpu_gpu_oversubscription() -> None:
    controls = _controls()
    assert controls.as_dict() == {
        "inner_fold_parallelism": 4,
        "fold_parallel_backend": "processes",
        "fold_slots_per_device": 2,
        "bank_parallelism": 2,
        "worker_cpu_threads": 1,
        "schema_version": (
            ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA
        ),
    }
    with pytest.raises(ValueError, match="every field exactly"):
        RoleNeutralNeuralQueryOperationalControls.from_mapping(
            {
                key: value
                for key, value in controls.as_dict().items()
                if key != "bank_parallelism"
            }
        )
    with pytest.raises(ValueError, match="per-device slots"):
        _controls(inner_fold_parallelism=5).bind_task_resources(
            devices=("cuda:0", "cuda:1"),
            owner_cpu_budget=8,
        )
    with pytest.raises(ValueError, match="global CPU lease"):
        controls.bind_task_resources(
            devices=("cuda:0", "cuda:1"),
            owner_cpu_budget=3,
        )
    with pytest.raises(ValueError, match="spawned processes"):
        _controls(fold_parallel_backend="threads").bind_task_resources(
            devices=("cuda:0", "cuda:1"),
            owner_cpu_budget=4,
        )
    with pytest.raises(ValueError, match="must be one"):
        _controls(worker_cpu_threads=2)


def test_cuda_child_records_allocated_and_reserved_peaks_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda device: calls.append(("set", str(device))),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", str(device))),
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: calls.append(("sync", str(device))),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device: 123,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda device: 456,
    )
    completed = task_execution_module._invoke_neural_query_task(
        lambda task, device: (task, device),
        "payload",
        "cuda:7",
        worker_cpu_threads=1,
        process_isolated=False,
    )
    assert completed.value == ("payload", "cuda:7")
    assert completed.gpu_peak_allocated_bytes == 123
    assert completed.gpu_peak_reserved_bytes == 456
    assert calls == [
        ("set", "cuda:7"),
        ("reset", "cuda:7"),
        ("sync", "cuda:7"),
    ]


def test_spawned_tasks_overlap_share_slots_use_both_gpus_and_merge_canonically(
    tmp_path: Path,
) -> None:
    multi_gpu_plan = _controls(
        inner_fold_parallelism=2,
        fold_slots_per_device=1,
        bank_parallelism=2,
    ).bind_task_resources(
        devices=("cuda:0", "cuda:1"),
        owner_cpu_budget=2,
    )
    multi_root = tmp_path / "multi"
    multi_root.mkdir()
    multi_tasks = tuple(
        {
            "index": index,
            "delay": 0.18 - 0.06 * index,
            "barrier_root": str(multi_root),
            "expected": 2,
        }
        for index in range(2)
    )
    multi_values, multi_attestation = execute_bounded_neural_query_tasks(
        multi_tasks,
        task_names=("fold_001", "fold_002"),
        resource_plan=multi_gpu_plan,
        worker=_barrier_echo,
        parallelism=multi_gpu_plan.inner_fold_parallelism,
        phase="test_multi_gpu_inner_folds",
    )

    assert [row["index"] for row in multi_values] == [0, 1]
    assert [row["device_seen"] for row in multi_values] == [
        "cuda:0",
        "cuda:1",
    ]
    assert multi_attestation["maximum_concurrent_leases"] == 2
    assert {
        row["device"] for row in multi_attestation["task_intervals"]
    } == {"cuda:0", "cuda:1"}
    assert len(
        {
            row["process_id"]
            for row in multi_attestation["task_intervals"]
        }
    ) == 2
    assert multi_attestation["canonical_result_order_restored"] is True

    shared_gpu_plan = _controls(
        inner_fold_parallelism=2,
        fold_slots_per_device=2,
        bank_parallelism=2,
    ).bind_task_resources(
        devices=("cuda:0",),
        owner_cpu_budget=2,
    )
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    shared_tasks = tuple(
        {
            "index": index,
            "delay": 0.18 - 0.06 * index,
            "barrier_root": str(shared_root),
            "expected": 2,
        }
        for index in range(2)
    )
    shared_values, shared_attestation = (
        execute_bounded_neural_query_tasks(
            shared_tasks,
            task_names=("fold_001", "fold_002"),
            resource_plan=shared_gpu_plan,
            worker=_barrier_echo,
            parallelism=shared_gpu_plan.inner_fold_parallelism,
            phase="test_shared_gpu_inner_folds",
        )
    )
    assert [row["index"] for row in shared_values] == [0, 1]
    assert [row["device_seen"] for row in shared_values] == [
        "cuda:0",
        "cuda:0",
    ]
    assert shared_attestation["maximum_concurrent_leases"] == 2
    assert (
        shared_attestation["per_device"]["cuda:0"][
            "maximum_concurrent_leases"
        ]
        == 2
    )
    assert len(
        {
            row["process_id"]
            for row in shared_attestation["task_intervals"]
        }
    ) == 2
    assert shared_attestation["canonical_result_order_restored"] is True


def _write_embedding_cache(
    root: Path,
    *,
    texts: tuple[str, ...],
) -> None:
    root.mkdir()
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    with (root / "chunk_embeddings.npy").open("xb") as handle:
        np.save(handle, embeddings)
    with (root / "offsets.npy").open("xb") as handle:
        np.save(
            handle,
            np.arange(len(texts) + 1, dtype=np.int64),
        )
    metadata = {
        "num_samples": len(texts),
        "hidden_size": embeddings.shape[1],
        "chunk_size_words": 100,
        "chunk_overlap_words": 0,
        "max_chunks": 10,
        "chunk_selection": "last",
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "chunk_texts.jsonl").write_text(
        "".join(
            json.dumps({"chunks": [text]}, sort_keys=True) + "\n"
            for text in texts
        ),
        encoding="utf-8",
    )


def test_authenticated_mmap_reference_reopens_rows_and_refuses_peer_access(
    tmp_path: Path,
) -> None:
    texts = (
        "authorized alpha row",
        "authorized beta row",
        "peer sentinel must not enter the task payload",
    )
    cache_root = tmp_path / "cache"
    _write_embedding_cache(cache_root, texts=texts)
    cache = SpentOnlyFrozenChunkEmbeddingCache(cache_root)
    bound = cache.bind_spent((0, 1), texts[:2])
    reference = NeuralQueryAuthenticatedCacheReference.from_bound_provider(
        bound,
        allowed_row_ids=(0, 1),
    )

    payload = pickle.dumps(reference)
    assert texts[2].encode("utf-8") not in payload
    reopened = reference.open_bound(
        row_ids=(1, 0),
        texts=(texts[1], texts[0]),
    )
    assert isinstance(reopened._embeddings, np.memmap)
    np.testing.assert_array_equal(
        reopened.chunk_matrix(1),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    )
    with pytest.raises(PermissionError, match="peer-row access"):
        reference.open_bound(
            row_ids=(0, 2),
            texts=(texts[0], texts[2]),
        )


def test_service_aggregates_self_hashed_operational_phases() -> None:
    resource_plan = _controls(
        inner_fold_parallelism=2,
        fold_slots_per_device=2,
        bank_parallelism=2,
    ).bind_task_resources(
        devices=("cpu",),
        owner_cpu_budget=2,
    )
    discovery_body = {
        "schema_version": (
            "production_neural_query_discovery_execution_attestation_v1"
        ),
        "resource_plan": resource_plan.as_dict(),
    }
    safe_body = {
        "schema_version": (
            "production_neural_query_task_phase_execution_attestation_v1"
        ),
        "phase": "safe_evidence_banks",
    }
    heldout_body = {
        "schema_version": (
            "production_neural_query_task_phase_execution_attestation_v1"
        ),
        "phase": "heldout_moment_banks",
    }
    service = object.__new__(ContextFitNeuralQueryService)
    service._task_resource_plan = resource_plan
    service._operational_attestations = [
        {
            **body,
            "content_sha256": _canonical_sha256(body),
        }
        for body in (discovery_body, safe_body, heldout_body)
    ]

    aggregate = service.operational_attestation()
    assert aggregate["phase_order"] == [
        "inner_folds_then_consensus_final_refits",
        "safe_evidence_banks",
        "heldout_moment_banks",
    ]
    assert aggregate["phase_count"] == 3
    assert aggregate["scientific_payload_contains_device_metadata"] is False
    assert aggregate["attestation_embedded_in_scientific_artifact"] is False
    assert aggregate["content_sha256"] == _canonical_sha256(
        {
            key: value
            for key, value in aggregate.items()
            if key != "content_sha256"
        }
    )


def _real_cache_service(
    tmp_path: Path,
    *,
    cache: SpentOnlyFrozenChunkEmbeddingCache,
    resource_plan,
) -> ContextFitNeuralQueryService:
    service = object.__new__(ContextFitNeuralQueryService)
    service.cache_dir = tmp_path / "service-cache"
    service.cache_dir.mkdir()
    service._owned_discoveries = {}
    service._owned_discovery_bindings = {}
    service._owned_discovery_content_sha256s = {}
    service.dataset_path = tmp_path / "dataset.parquet"
    service.dataset_path.write_bytes(b"test")
    service.stage1_config_path = tmp_path / "stage1.json"
    service.stage1_config_path.write_text("{}\n", encoding="utf-8")
    service._stage1_config_snapshot = SimpleNamespace(
        sha256=hashlib.sha256(b"{}\n").hexdigest(),
        verify_source=lambda: None,
    )
    service.text_column = "clinical_text"
    service.embedding_cache = cache
    service._dataset_row_count = cache.row_count
    service._nuisance_views = ({"name": "test"},)
    service._nuisance_stack_config = TfidfNuisanceStackScientificConfig()
    service.query_config = NeuralQueryAgenticForestConfig(
        treatment_query_count=1,
        outcome_query_count=1,
        effect_query_count=1,
        query_inner_folds=2,
        initial_pool_size=1,
        query_epochs=1,
        final_refit_epochs=1,
        evidence_top_patients=1,
        evidence_background_patients=1,
        evidence_top_ngrams=2,
        max_features_per_query=1,
        max_raw_feature_candidates=3,
        max_canonical_features=3,
    )
    service.query_config.validate()
    service.nuisance_folds = 2
    service.devices = ("cpu",)
    service._task_resource_plan = resource_plan
    service._operational_attestations = []
    service.seed = 17
    service.outcome_type = "binary"
    service._identity = service._identity_payload()
    return service


def test_safe_evidence_and_moment_banks_share_bounded_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = (
        "alpha treatment marker",
        "beta outcome marker",
        "gamma effect marker",
    )
    cache_root = tmp_path / "cache"
    _write_embedding_cache(cache_root, texts=texts)
    cache = SpentOnlyFrozenChunkEmbeddingCache(cache_root)
    resource_plan = _controls(
        inner_fold_parallelism=2,
        fold_parallel_backend="threads",
        fold_slots_per_device=2,
        bank_parallelism=2,
    ).bind_task_resources(
        devices=("cpu",),
        owner_cpu_budget=2,
    )
    service = _real_cache_service(
        tmp_path,
        cache=cache,
        resource_plan=resource_plan,
    )
    discovery = {
        "banks": {
            bank: {
                "queries": np.asarray(
                    [[1.0, 0.0]],
                    dtype=np.float32,
                ),
                "records": [
                    {
                        "query_id": f"{bank}_context_query_001",
                        "member_count": 2,
                        "fit_standardized_score": 0.4,
                    }
                ],
            }
            for bank in ("treatment", "outcome", "effect")
        }
    }

    def fake_build_query_evidence(*, bank, **_kwargs):
        return [
            {
                "query_id": f"{bank}_context_query_001",
                "bank": bank,
                "mechanical_role": f"{bank}_role",
                "member_count": 2,
                "fit_standardized_score": 0.4,
                "top_contrastive_ngrams": [
                    {"term": f"{bank} marker", "rank": 1}
                ],
            }
        ]

    original_safe = context_module._run_safe_evidence_bank
    original_moment = context_module._run_moment_bank

    def delayed_safe(task, device):
        time.sleep(0.08)
        return original_safe(task, device)

    def delayed_moment(task, device):
        time.sleep(0.08)
        return original_moment(task, device)

    monkeypatch.setattr(
        context_module,
        "build_query_evidence",
        fake_build_query_evidence,
    )
    monkeypatch.setattr(
        context_module,
        "_run_safe_evidence_bank",
        delayed_safe,
    )
    monkeypatch.setattr(
        context_module,
        "_run_moment_bank",
        delayed_moment,
    )

    evidence = service.safe_evidence(
        discovery=discovery,
        context_row_ids=(0, 1, 2),
        context_texts=texts,
    )
    names, kinds, roles, values = service.moments_for_rows(
        banks=discovery["banks"],
        row_ids=(0, 1, 2),
        texts=texts,
        row_name="heldout_rows",
        text_name="heldout_texts",
    )

    assert [row["bank"] for row in evidence] == [
        "treatment",
        "outcome",
        "effect",
    ]
    assert names == (
        "neural_query_treatment_signed_mean",
        "neural_query_treatment_absolute_max",
        "neural_query_treatment_signed_order_01",
        "neural_query_outcome_signed_mean",
        "neural_query_outcome_absolute_max",
        "neural_query_outcome_signed_order_01",
        "neural_query_effect_signed_mean",
        "neural_query_effect_absolute_max",
        "neural_query_effect_signed_order_01",
    )
    assert len(kinds) == len(roles) == values.shape[1] == len(names)
    assert values.shape == (3, 9)
    phases = service.operational_attestations()
    assert [row["phase"] for row in phases] == [
        "safe_evidence_banks",
        "heldout_moment_banks",
    ]
    assert all(
        row["maximum_concurrent_leases"] == 2 for row in phases
    )
