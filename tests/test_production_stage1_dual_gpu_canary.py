from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import oci.inference.production_stage1_dual_gpu_canary as canary
from oci.inference.production_stage1_dual_gpu_canary import (
    CANARY_RESOURCE_LEDGER_NAME,
    CANARY_TERMINAL_MANIFEST_NAME,
    STAGE1_DUAL_GPU_CANARY_TEST_DESCRIPTOR_SCHEMA,
    Stage1DualGpuCanaryOptions,
    parse_stage1_dual_gpu_canary_args,
    run_stage1_dual_gpu_canary,
    validate_stage1_dual_gpu_canary,
)


_FAKE_WORKER_TARGET = (
    "oci.inference.production_stage1_dual_gpu_canary:"
    "_cpu_fake_canary_worker"
)


def _write_test_descriptor(
    path: Path,
    *,
    configured_gpu_id: int = 2,
) -> Path:
    scope = {
        "canonical_index": 0,
        "scope_id": "outer_001_full",
        "scope_kind": "full_outer",
        "outer_fold": 1,
        "inner_fold": None,
        "context_epoch": None,
        "provider_inner_fold": None,
        "fit_row_ids": [0, 1, 2, 3],
        "heldout_row_ids": [4],
        "fit_row_count": 4,
        "heldout_row_count": 1,
        "fit_row_order_fingerprint": "a" * 64,
        "heldout_row_order_fingerprint": "b" * 64,
        "global_seed": 42,
        "scope_seed": 1234567,
        "heldout_labels_supplied": False,
        "scope_sha256": "c" * 64,
    }
    body = {
        "schema_version": STAGE1_DUAL_GPU_CANARY_TEST_DESCRIPTOR_SCHEMA,
        "stage1_request_sha256": "d" * 64,
        "plan_content_sha256": "e" * 64,
        "scope": scope,
        "assignment": {
            "scope_id": "outer_001_full",
            "gpu_id": configured_gpu_id,
            "execution_rank": 0,
            "fit_row_count": 4,
            "assigned_gpu_load_after": 4,
        },
    }
    path.write_text(
        json.dumps(
            {**body, "content_sha256": canary._sha256_json(body)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path.resolve(strict=True)


def _snapshot_binding(tmp_path: Path, *, identity: str = "f" * 64) -> dict:
    root = (tmp_path / "snapshot").resolve()
    root.mkdir(exist_ok=True)
    manifest = root / "source_snapshot_manifest.json"
    manifest.touch(exist_ok=True)
    return {
        "root": str(root),
        "manifest_path": str(manifest),
        "content_sha256": identity,
        "file_count": 7,
    }


def _resource_sampler(
    gpu_ids: tuple[int, ...],
    *,
    child_pids: tuple[int, ...],
) -> dict:
    del child_pids
    return {
        "sampled_at": "2026-07-23T12:00:00.000000Z",
        "gpus": [
            {
                "gpu_id": int(gpu_id),
                "uuid": f"GPU-{gpu_id}",
                "total_mib": 49140,
                "used_mib": 256,
                "free_mib": 48884,
                "utilization_percent": 0,
            }
            for gpu_id in gpu_ids
        ],
        "compute_apps": [],
    }


def _options(
    tmp_path: Path,
    *,
    fail_gpu_id: int | None = None,
    sleep_seconds: float = 0.2,
    minimum_concurrency_factor: float = 0.05,
    descendant_sentinel_path: Path | None = None,
    configured_gpu_id: int = 2,
) -> Stage1DualGpuCanaryOptions:
    tmp_path.mkdir(parents=True, exist_ok=True)
    descriptor = _write_test_descriptor(
        tmp_path / "descriptor_manifest.json",
        configured_gpu_id=configured_gpu_id,
    )
    snapshot = _snapshot_binding(tmp_path)
    return Stage1DualGpuCanaryOptions(
        descriptor_manifest_path=descriptor,
        source_snapshot_root=Path(snapshot["root"]),
        output_root=(tmp_path / "canary").resolve(),
        gpu_ids=(2, 3),
        resource_poll_seconds=0.05,
        maximum_reservation_fraction=0.85,
        minimum_headroom_bytes=6 * 1024**3,
        minimum_concurrency_factor=minimum_concurrency_factor,
        worker_target=_FAKE_WORKER_TARGET,
        production_worker_required=False,
        require_source_snapshot_execution=False,
        test_sleep_seconds=sleep_seconds,
        test_fail_gpu_id=fail_gpu_id,
        test_descendant_sentinel_path=descendant_sentinel_path,
    )


def _patch_snapshot(monkeypatch: pytest.MonkeyPatch, binding: dict) -> None:
    monkeypatch.setattr(
        canary,
        "_validate_snapshot_binding",
        lambda _options: dict(binding),
    )


def test_cpu_replicas_share_scientific_request_and_authenticate_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    binding = {
        "root": str(options.source_snapshot_root),
        "manifest_path": str(
            options.source_snapshot_root / "source_snapshot_manifest.json"
        ),
        "content_sha256": "f" * 64,
        "file_count": 7,
    }
    _patch_snapshot(monkeypatch, binding)

    result = run_stage1_dual_gpu_canary(
        options,
        resource_sampler=_resource_sampler,
    )

    assert result["status"] == "accepted"
    assert result["scientific_equality"] is True
    assert result["scientific_identity"]["content_sha256"]
    assert {
        replica["scientific_identity_content_sha256"]
        for replica in result["replicas"]
    } == {result["scientific_identity"]["content_sha256"]}
    global_terminal = options.output_root / CANARY_TERMINAL_MANIFEST_NAME
    assert global_terminal.stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns
        for path in options.output_root.rglob("*")
        if path.is_file() and path != global_terminal
    )
    request = json.loads(
        (options.output_root / "canary_request.json").read_text(encoding="utf-8")
    )
    assert "gpu_ids" not in request["scientific_request"]
    assert "configured_assignment" not in request["scientific_request"]
    assert "replica_logical_gpu_id" not in request["scientific_request"]
    assert request["descriptor"]["configured_assignment"]["gpu_id"] == 2
    assert request["descriptor"]["replica_logical_gpu_id"] == 0
    replica_manifests = [
        json.loads(
            (
                options.output_root
                / "replicas"
                / f"gpu_{gpu_id:03d}"
                / "replica_manifest.json"
            ).read_text(encoding="utf-8")
        )
        for gpu_id in (2, 3)
    ]
    assert {
        row["scientific_request_sha256"] for row in replica_manifests
    } == {request["scientific_request_sha256"]}
    assert {row["physical_gpu_id"] for row in replica_manifests} == {2, 3}
    assert {row["logical_gpu_id"] for row in replica_manifests} == {0}
    for gpu_id in (2, 3):
        replica_root = (
            options.output_root / "replicas" / f"gpu_{gpu_id:03d}"
        )
        terminal = replica_root / "replica_manifest.json"
        assert terminal.stat().st_mtime_ns >= max(
            path.stat().st_mtime_ns
            for path in replica_root.rglob("*")
            if path.is_file() and path != terminal
        )
    assert validate_stage1_dual_gpu_canary(
        options.output_root,
        allow_test_worker=True,
        require_source_snapshot_execution=False,
    ) == result


def test_configured_assignment_is_authenticated_but_runtime_gpu_is_replica_local(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path)
    descriptor = canary._descriptor_binding(options)
    snapshot = _snapshot_binding(tmp_path, identity="f" * 64)
    request = canary._build_request(
        options=options,
        snapshot=snapshot,
        descriptor=descriptor,
    )
    replica = canary._replica_request(
        root=(tmp_path / "replica").resolve(),
        physical_gpu_id=2,
        canary_request=request,
        descriptor=descriptor,
        options=options,
    )

    assert replica.assignment["gpu_id"] == 2
    assert replica.gpu_id == 0
    assert descriptor.as_dict()["configured_assignment"]["gpu_id"] == 2
    assert descriptor.as_dict()["replica_logical_gpu_id"] == 0

    mismatched = _options(
        tmp_path / "mismatch",
        configured_gpu_id=0,
    )
    with pytest.raises(ValueError, match="configured canary GPU inventory"):
        canary._descriptor_binding(mismatched)


def test_payload_descriptor_and_extra_file_tampering_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    binding = {
        "root": str(options.source_snapshot_root),
        "manifest_path": str(
            options.source_snapshot_root / "source_snapshot_manifest.json"
        ),
        "content_sha256": "f" * 64,
        "file_count": 7,
    }
    _patch_snapshot(monkeypatch, binding)
    run_stage1_dual_gpu_canary(options, resource_sampler=_resource_sampler)

    proof = (
        options.output_root
        / "replicas"
        / "gpu_002"
        / "payload"
        / "fake_scientific_proof.json"
    )
    original = proof.read_bytes()
    proof.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="payload changed"):
        validate_stage1_dual_gpu_canary(
            options.output_root,
            allow_test_worker=True,
            require_source_snapshot_execution=False,
        )
    proof.write_bytes(original)

    options.descriptor_manifest_path.write_bytes(
        options.descriptor_manifest_path.read_bytes() + b"\n"
    )
    with pytest.raises(ValueError, match="descriptor changed|request does not"):
        validate_stage1_dual_gpu_canary(
            options.output_root,
            allow_test_worker=True,
            require_source_snapshot_execution=False,
        )

    # Restore the external descriptor before testing closed output-tree checks.
    options.descriptor_manifest_path.write_bytes(
        options.descriptor_manifest_path.read_bytes().rstrip() + b"\n"
    )
    (options.output_root / "unregistered.txt").write_text(
        "tamper",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        validate_stage1_dual_gpu_canary(
            options.output_root,
            allow_test_worker=True,
            require_source_snapshot_execution=False,
        )


def test_peer_failure_and_parent_cancellation_terminate_and_join_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descendant_sentinel = tmp_path / "orphaned_descendant.txt"
    failing = _options(
        tmp_path / "failure_case",
        fail_gpu_id=2,
        sleep_seconds=5.0,
        descendant_sentinel_path=descendant_sentinel,
    )
    binding = {
        "root": str(failing.source_snapshot_root),
        "manifest_path": str(
            failing.source_snapshot_root / "source_snapshot_manifest.json"
        ),
        "content_sha256": "f" * 64,
        "file_count": 7,
    }
    _patch_snapshot(monkeypatch, binding)
    with pytest.raises(RuntimeError, match="replica failed|exited with code"):
        run_stage1_dual_gpu_canary(
            failing,
            resource_sampler=_resource_sampler,
        )
    assert not (failing.output_root / CANARY_TERMINAL_MANIFEST_NAME).exists()
    assert json.loads(
        (failing.output_root / CANARY_RESOURCE_LEDGER_NAME).read_text(
            encoding="utf-8"
        )
    )["status"] == "failed"
    assert not [
        child
        for child in mp.active_children()
        if child.name.startswith("stage1-canary-gpu-")
    ]
    time.sleep(2.0)
    assert not descendant_sentinel.exists()

    cancelled_root = tmp_path / "cancelled_case"
    cancelled = _options(cancelled_root, sleep_seconds=5.0)
    cancelled_binding = {
        "root": str(cancelled.source_snapshot_root),
        "manifest_path": str(
            cancelled.source_snapshot_root / "source_snapshot_manifest.json"
        ),
        "content_sha256": "f" * 64,
        "file_count": 7,
    }
    _patch_snapshot(monkeypatch, cancelled_binding)
    cancellation = threading.Event()
    cancellation.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        run_stage1_dual_gpu_canary(
            cancelled,
            resource_sampler=_resource_sampler,
            cancellation_event=cancellation,
        )
    assert not (cancelled.output_root / CANARY_TERMINAL_MANIFEST_NAME).exists()
    assert not [
        child
        for child in mp.active_children()
        if child.name.startswith("stage1-canary-gpu-")
    ]


def test_resource_contract_failure_is_not_published_as_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    binding = {
        "root": str(options.source_snapshot_root),
        "manifest_path": str(
            options.source_snapshot_root / "source_snapshot_manifest.json"
        ),
        "content_sha256": "f" * 64,
        "file_count": 7,
    }
    _patch_snapshot(monkeypatch, binding)

    def high_use(gpu_ids, *, child_pids):
        sample = _resource_sampler(tuple(gpu_ids), child_pids=tuple(child_pids))
        for row in sample["gpus"]:
            row["used_mib"] = 45000
            row["free_mib"] = 4140
        return sample

    with pytest.raises(RuntimeError, match="physically idle|memory/headroom"):
        run_stage1_dual_gpu_canary(options, resource_sampler=high_use)
    assert not (options.output_root / CANARY_TERMINAL_MANIFEST_NAME).exists()


def test_snapshot_execution_binding_and_descriptor_bytes_enter_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root = (tmp_path / "snapshot").resolve()
    snapshot_root.mkdir()
    fake = SimpleNamespace(
        root=snapshot_root,
        manifest_path=snapshot_root / "source_snapshot_manifest.json",
        content_sha256="1" * 64,
        file_count=12,
        as_dict=lambda: {
            "root": str(snapshot_root),
            "manifest_path": str(
                snapshot_root / "source_snapshot_manifest.json"
            ),
            "content_sha256": "1" * 64,
            "file_count": 12,
        },
    )
    monkeypatch.setattr(
        canary,
        "validate_production_source_snapshot",
        lambda _root: fake,
    )
    options = Stage1DualGpuCanaryOptions(
        descriptor_manifest_path=(tmp_path / "unused.json").resolve(),
        source_snapshot_root=snapshot_root,
        output_root=(tmp_path / "output").resolve(),
        gpu_ids=(2, 3),
        production_worker_required=False,
        require_source_snapshot_execution=True,
        worker_target=_FAKE_WORKER_TARGET,
    )
    monkeypatch.setattr(
        canary,
        "__file__",
        str(snapshot_root / "oci" / "inference" / "module.py"),
    )
    monkeypatch.setenv(canary.SOURCE_SNAPSHOT_EXECUTION_ENV, "1" * 64)
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    assert canary._validate_snapshot_binding(options)["content_sha256"] == "1" * 64
    monkeypatch.setenv(canary.SOURCE_SNAPSHOT_EXECUTION_ENV, "2" * 64)
    with pytest.raises(ValueError, match="authenticated source snapshot"):
        canary._validate_snapshot_binding(options)


def test_cli_parsing_binds_two_gpus_and_production_thresholds(
    tmp_path: Path,
) -> None:
    options = parse_stage1_dual_gpu_canary_args(
        [
            "--descriptor-manifest",
            str(tmp_path / "descriptor.json"),
            "--source-snapshot-root",
            str(tmp_path / "snapshot"),
            "--output-root",
            str(tmp_path / "output"),
            "--scope-id",
            "outer_003_full",
            "--gpu-id",
            "1",
            "--gpu-id",
            "0",
        ]
    )

    assert options.gpu_ids == (1, 0)
    assert options.scope_id == "outer_003_full"
    assert options.worker_target == canary.LEGACY_STAGE1_SCOPE_WORKER_TARGET
    assert options.production_worker_required is True
    assert options.maximum_reservation_fraction == 0.85
    assert options.minimum_headroom_bytes == 6 * 1024**3
    assert options.minimum_concurrency_factor == 1.5

    with pytest.raises(SystemExit):
        parse_stage1_dual_gpu_canary_args(
            [
                "--descriptor-manifest",
                str(tmp_path / "descriptor.json"),
                "--source-snapshot-root",
                str(tmp_path / "snapshot"),
                "--output-root",
                str(tmp_path / "output"),
                "--gpu-id",
                "0",
            ]
        )


def test_production_mode_cannot_switch_worker_or_reuse_output(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "output").resolve()
    options = Stage1DualGpuCanaryOptions(
        descriptor_manifest_path=(tmp_path / "descriptor.json").resolve(),
        source_snapshot_root=(tmp_path / "snapshot").resolve(),
        output_root=output,
        gpu_ids=(2, 3),
        worker_target=_FAKE_WORKER_TARGET,
        production_worker_required=True,
    )
    with pytest.raises(ValueError, match="cannot switch"):
        canary._validate_options(options)

    output.mkdir()
    correct_worker = Stage1DualGpuCanaryOptions(
        descriptor_manifest_path=options.descriptor_manifest_path,
        source_snapshot_root=options.source_snapshot_root,
        output_root=output,
        gpu_ids=(2, 3),
    )
    with pytest.raises(FileExistsError, match="fresh"):
        canary._validate_options(correct_worker)


def test_physical_uuid_mapping_exposes_one_logical_gpu_and_restores_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, str | None] = {}

    class _FakeProcess:
        def start(self) -> None:
            observed["visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")
            observed["order"] = os.environ.get("CUDA_DEVICE_ORDER")
            observed["hash_seed"] = os.environ.get("PYTHONHASHSEED")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-visible")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "parent-order")
    monkeypatch.setenv("PYTHONHASHSEED", "parent-seed")

    canary._start_canary_replica(
        _FakeProcess(),
        scope_seed=31415,
        physical_gpu_uuid="GPU-01234567-89ab-cdef-0123-456789abcdef",
        production_worker_required=True,
    )

    assert observed == {
        "visible": "GPU-01234567-89ab-cdef-0123-456789abcdef",
        "order": "PCI_BUS_ID",
        "hash_seed": "31415",
    }
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-visible"
    assert os.environ["CUDA_DEVICE_ORDER"] == "parent-order"
    assert os.environ["PYTHONHASHSEED"] == "parent-seed"


def test_snapshot_reexec_sets_and_authenticates_global_hash_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root = (tmp_path / "snapshot").resolve()
    entrypoint = snapshot_root / "scripts" / (
        "run_stage1_dual_gpu_reproducibility_canary.py"
    )
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("pass\n", encoding="utf-8")
    snapshot = SimpleNamespace(
        root=snapshot_root,
        content_sha256="9" * 64,
    )
    monkeypatch.setattr(
        canary,
        "validate_production_source_snapshot",
        lambda _root: snapshot,
    )
    captured = {}

    def fake_execve(executable, arguments, environment):
        captured.update(
            {
                "executable": executable,
                "arguments": arguments,
                "environment": environment,
            }
        )
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(canary.os, "execve", fake_execve)
    monkeypatch.delenv(canary.SOURCE_SNAPSHOT_EXECUTION_ENV, raising=False)
    with pytest.raises(RuntimeError, match="exec intercepted"):
        canary._reexec_from_source_snapshot(
            source_snapshot_root=snapshot_root,
            seed=71,
            raw_argv=("--seed", "71"),
        )
    assert captured["environment"]["PYTHONHASHSEED"] == "71"
    assert captured["environment"][canary.SOURCE_SNAPSHOT_EXECUTION_ENV] == "9" * 64

    monkeypatch.setattr(
        canary,
        "__file__",
        str(snapshot_root / "oci" / "inference" / "canary.py"),
    )
    monkeypatch.setenv(canary.SOURCE_SNAPSHOT_EXECUTION_ENV, "9" * 64)
    monkeypatch.setenv("PYTHONHASHSEED", "70")
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
        canary._reexec_from_source_snapshot(
            source_snapshot_root=snapshot_root,
            seed=71,
            raw_argv=("--seed", "71"),
        )
