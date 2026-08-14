from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_completed_handoff_uses_stage2_only_without_gpu_probe(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "output"
    handoff_dir = output_dir / "handoff"
    handoff_dir.mkdir(parents=True)
    (handoff_dir / "evidence.jsonl").write_text("{}\n", encoding="utf-8")
    (handoff_dir / "complete.json").write_text("{}\n", encoding="utf-8")

    invocation_log = tmp_path / "python_invocations.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "${FAKE_PYTHON_INVOCATION_LOG}"
printf '\n' >> "${FAKE_PYTHON_INVOCATION_LOG}"
if [[ "${1:-}" == *detect_all_evidence_hardware.py ]]; then
    printf '0\tcpu\t12\t32\t12\tnot inspected (endpoint-backed Stage 2 only)\n'
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_PYTHON_INVOCATION_LOG": str(invocation_log),
            "GPU_COUNT": "not-a-number",
            "OCI_PYTHON": str(fake_python),
            "PHYSICAL_GPUS": "also-not-a-number",
            "STAGE2_ENDPOINT": "http://stage2.test/v1",
            "STAGE2_WORKERS": "",
            "STAGE2_VLLM_SERVERS": "0",
        }
    )
    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "run_synthetic_all_evidence.sh"),
            (
                "synthetic_data/example_synthetic_datasets/"
                "five_confounders_five_effect_modifiers_nsclc_with_structured/"
                "dataset.parquet"
            ),
            "test_output",
            str(output_dir),
        ],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 2
    assert "detect_all_evidence_hardware.py" in invocations[0]
    assert "--stage2-only" in invocations[0]
    assert "--stage2-workers 32" in invocations[0]
    assert "-c" not in invocations[0].split()
    assert "research_all_evidence_workflow" in invocations[1]
    assert "--stage2-only" in invocations[1]
    assert "--devices cpu" in invocations[1]
    assert "stage2.workers=32" in invocations[1]
    assert "CUDA devices:   not required for endpoint-backed Stage 2" in completed.stdout
    assert "HTR modeling:   not run during Stage 2-only resume" in completed.stdout


def test_managed_vllm_keeps_gpu_detection_and_pool_arguments(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "output"
    invocation_log = tmp_path / "python_invocations.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "${FAKE_PYTHON_INVOCATION_LOG}"
printf '\n' >> "${FAKE_PYTHON_INVOCATION_LOG}"
if [[ "${1:-}" == *detect_all_evidence_hardware.py ]]; then
    printf '2\tcuda:0,cuda:1\t12\t32\t12\ttwo eligible GPUs\n'
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_PYTHON_INVOCATION_LOG": str(invocation_log),
            "GPU_COUNT": "2",
            "OCI_PYTHON": str(fake_python),
            "PHYSICAL_GPUS": "",
            "STAGE2_ENDPOINT": "",
            "STAGE2_MODEL": "Qwen/Qwen3-32B",
            "STAGE2_VLLM_DOWNLOAD_DIR": "",
            "STAGE2_VLLM_EXTRA_ARGS_JSON": "",
            "STAGE2_VLLM_GPUS": "",
            "STAGE2_VLLM_SERVERS": "2",
            "STAGE2_WORKERS": "",
        }
    )
    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "run_synthetic_all_evidence.sh"),
            (
                "synthetic_data/example_synthetic_datasets/"
                "five_confounders_five_effect_modifiers_nsclc_with_structured/"
                "dataset.parquet"
            ),
            "test_output",
            str(output_dir),
        ],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 4
    assert "sentence_transformers" in invocations[0]
    assert "find_spec" in invocations[1] and "vllm" in invocations[1]
    assert "detect_all_evidence_hardware.py" in invocations[2]
    assert "--gpu-count 2" in invocations[2]
    assert "--stage2-workers 32" in invocations[2]
    assert "--stage2-only" not in invocations[2]
    assert "research_all_evidence_workflow" in invocations[3]
    assert "--stage2-vllm-servers 2" in invocations[3]
    assert r"--stage2-vllm-gpus cuda:0\,cuda:1" in invocations[3]
    assert "--stage2-model Qwen/Qwen3-32B" in invocations[3]
    assert "stage2.workers=32" in invocations[3]
    assert "--stage2-only" not in invocations[3]
    assert "Stage 2:        managed vLLM: 2 servers on cuda:0,cuda:1" in completed.stdout
