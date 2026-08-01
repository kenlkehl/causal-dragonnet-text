from __future__ import annotations

from types import SimpleNamespace

import pytest

import oci.inference.performance_telemetry as telemetry
import oci.inference.portable_resource_scheduler as scheduler
import oci.inference.production_all_evidence_workflow as workflow


def test_numeric_cuda_visibility_maps_nvml_to_logical_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(argv, **_kwargs):
        query = argv[1]
        if query == (
            "--query-gpu=index,uuid,memory.total,memory.free,"
            "utilization.gpu"
        ):
            return SimpleNamespace(
                stdout=(
                    "0, uuid-0, 100, 90, 10\n"
                    "2, uuid-2, 200, 150, 20\n"
                )
            )
        if query == "--query-compute-apps=gpu_uuid,pid,used_gpu_memory":
            return SimpleNamespace(stdout="uuid-2, 123, 7\n")
        if query == "--query-compute-apps=gpu_uuid,pid,used_memory":
            return SimpleNamespace(
                stdout=(
                    "uuid-0, 111, 3\n"
                    "uuid-2, 222, 9\n"
                )
            )
        if query == (
            "--query-gpu=index,uuid,utilization.gpu,memory.used,"
            "memory.total"
        ):
            return SimpleNamespace(
                stdout=(
                    "0, uuid-0, 10, 10, 100\n"
                    "2, uuid-2, 20, 50, 200\n"
                )
            )
        if query == (
            "--query-gpu=index,uuid,memory.total,memory.used,"
            "utilization.gpu"
        ):
            return SimpleNamespace(
                stdout=(
                    "0, uuid-0, 100, 10, 10\n"
                    "2, uuid-2, 200, 50, 20\n"
                )
            )
        raise AssertionError(f"unexpected nvidia-smi query: {query}")

    monkeypatch.setattr(scheduler.subprocess, "run", run)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")

    inventory = scheduler.discover_resources()
    assert [(gpu.device, gpu.uuid) for gpu in inventory.gpus] == [
        ("cuda:0", "uuid-2"),
        ("cuda:1", "uuid-0"),
    ]
    assert inventory.gpus[0].external_processes == (
        {"pid": 123, "used_memory_bytes": 7 * 1024 * 1024},
    )
    samples = telemetry.sample_nvidia_gpus(("cuda:0", "cuda:1"))
    assert [(row["device"], row["uuid"]) for row in samples] == [
        ("cuda:0", "uuid-2"),
        ("cuda:1", "uuid-0"),
    ]
    safety = SimpleNamespace(
        gpu_max_allocation_fraction=0.9,
        gpu_minimum_headroom_bytes=1,
        fail_on_external_gpu_occupants=False,
        content_sha256="0" * 64,
    )
    safety.as_dict = lambda: {
        "gpu_max_allocation_fraction": (
            safety.gpu_max_allocation_fraction
        ),
        "gpu_minimum_headroom_bytes": safety.gpu_minimum_headroom_bytes,
        "fail_on_external_gpu_occupants": (
            safety.fail_on_external_gpu_occupants
        ),
    }
    subject = SimpleNamespace(
        stage1_gpu_ids=(0,),
        options=SimpleNamespace(resource_performance_safety=safety),
    )
    preflight = workflow.ProductionAllEvidenceWorkflow._gpu_preflight(
        subject
    )
    assert preflight["gpu_uuids"] == {"0": "uuid-2"}
    assert preflight["observed_compute_processes"] == {
        0: [{"pid": 222, "used_memory_mib": "9"}],
    }

    safety.gpu_max_allocation_fraction = 0.25
    workflow.ProductionAllEvidenceWorkflow._gpu_preflight(subject)

    safety.gpu_max_allocation_fraction = 0.249
    with pytest.raises(RuntimeError) as unsafe:
        workflow.ProductionAllEvidenceWorkflow._gpu_preflight(subject)
    assert '"0":{"allocation_fraction":0.25' in str(unsafe.value)
    assert '"uuid":"uuid-2"' in str(unsafe.value)

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert [gpu.device for gpu in scheduler.discover_resources().gpus] == [
        "cuda:0",
        "cuda:2",
    ]

    for unsupported in ("GPU-uuid-2", "MIG-instance"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", unsupported)
        with pytest.raises(ValueError, match="UUID and MIG masks"):
            scheduler.discover_resources()
        with pytest.raises(ValueError, match="UUID and MIG masks"):
            telemetry.sample_nvidia_gpus(("cuda:0",))
