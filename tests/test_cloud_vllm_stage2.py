from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

from oci.inference.portable_workflow_spec import ScientificWorkflowSpec
from oci.inference.portable_artifacts import (
    ArtifactCompatibility,
    publish_portable_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "nvidia/Gemma-4-31B-IT-NVFP4"
MODEL_REVISION = "4135a98a9b728a548947683219633b25682223ac"
SERVED_MODEL = f"{MODEL_ID}@{MODEL_REVISION}"


def _module(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _model_metadata(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "text_config": {"max_position_embeddings": 262_144},
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt"},
                "quantization": {"quant_algo": "NVFP4"},
            }
        ),
        encoding="utf-8",
    )


def _free_port() -> int:
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def test_full_stage2_model_materialization_rejects_non_nvfp4_tree(
    tmp_path: Path,
) -> None:
    materializer = _module(
        "scripts/materialize_production_models.py",
        "test_materialize_production_models",
    )
    target = tmp_path / "model"
    _model_metadata(target)
    for name in materializer.REQUIRED_FILES["stage2_vllm"]:
        path = target / name
        if not path.exists():
            path.write_bytes(b"fixture")
    marker = materializer._closed_marker(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        resolved_revision=MODEL_REVISION,
        kind="stage2_vllm",
    )
    (target / materializer.MARKER_NAME).write_text(
        json.dumps(marker), encoding="utf-8"
    )

    materializer._validate_tree(
        target,
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        kind="stage2_vllm",
    )

    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"]["quant_algo"] = "FP8"
    (target / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="ModelOpt NVFP4"):
        materializer._validate_tree(
            target,
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            kind="stage2_vllm",
        )


def test_lazy_proxy_starts_exact_eight_gpu_vllm_only_on_stage2_request(
    tmp_path: Path,
) -> None:
    proxy = _module(
        "scripts/run_local_vllm_stage2_proxy.py",
        "test_run_local_vllm_stage2_proxy",
    )
    model = tmp_path / "model"
    _model_metadata(model)
    fake_vllm = tmp_path / "vllm"
    fake_vllm.write_text(
        """#!/usr/bin/env python3
import json
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index('--port') + 1])
model = sys.argv[sys.argv.index('--served-model-name') + 1]

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        if self.path == '/health':
            body = b''
        elif self.path == '/v1/models':
            body = json.dumps({'data': [{'id': model}]}).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = HTTPServer(('127.0.0.1', port), Handler)
server.serve_forever()
""",
        encoding="utf-8",
    )
    fake_vllm.chmod(0o700)
    listen_port = _free_port()
    upstream_port = _free_port()
    while upstream_port == listen_port:
        upstream_port = _free_port()
    configuration = proxy.ProxyConfiguration(
        listen_host="127.0.0.1",
        listen_port=listen_port,
        upstream_host="127.0.0.1",
        upstream_port=upstream_port,
        vllm_command=fake_vllm,
        model_dir=model,
        served_model_name=SERVED_MODEL,
        tensor_parallel_size=8,
        max_model_len=262_144,
        gpu_memory_utilization=0.9,
        max_num_seqs=8,
        startup_timeout_seconds=10.0,
        request_timeout_seconds=10.0,
        log_path=tmp_path / "vllm.log",
        status_path=tmp_path / "status.json",
    )
    command = configuration.command()
    assert command[command.index("--quantization") + 1] == "modelopt"
    assert command[command.index("--tensor-parallel-size") + 1] == "8"
    assert command[command.index("--max-model-len") + 1] == "262144"
    assert command[command.index("--reasoning-parser") + 1] == "gemma4"
    assert command[command.index("--generation-config") + 1] == "vllm"

    supervisor = proxy.VllmSupervisor(configuration)
    server = proxy.LazyProxyServer(configuration, supervisor)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{listen_port}/proxy-health", timeout=2.0
        ) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["vllm_started"] is False
        assert supervisor.process is None

        with urllib.request.urlopen(
            f"http://127.0.0.1:{listen_port}/v1/models", timeout=15.0
        ) as response:
            models = json.loads(response.read().decode("utf-8"))
        assert models == {"data": [{"id": SERVED_MODEL}]}
        assert supervisor.process is not None
        assert supervisor.process.poll() is None
        status = json.loads(configuration.status_path.read_text(encoding="utf-8"))
        assert status["state"] == "vllm_ready"
        assert status["tensor_parallel_size"] == 8
        assert status["max_model_len"] == 262_144
    finally:
        server.shutdown()
        server.server_close()
        supervisor.shutdown()
        server_thread.join(timeout=5.0)
    assert supervisor.process is not None
    assert supervisor.process.poll() is not None


def test_cloud_scientific_profile_uses_full_context_without_truncation() -> None:
    profile = json.loads(
        (ROOT / "example_configs/portable_all_evidence_scientific_nsclc.json").read_text(
            encoding="utf-8"
        )
    )
    compiled = ScientificWorkflowSpec.from_mapping(profile)
    protocol = compiled.stage2_prompt_protocol
    assert protocol.model_context_window_tokens == 262_144
    assert protocol.max_rendered_discovery_prompt_bytes == 440_000
    assert protocol.proposal_max_tokens == 25_000
    assert protocol.extraction_max_tokens == 25_000


def test_cloud_cache_recovery_resolves_authenticated_relocation_inputs(
    tmp_path: Path,
) -> None:
    resolver = _module(
        "scripts/resolve_production_embedding_cache_import.py",
        "test_resolve_production_embedding_cache_import",
    )
    prior = tmp_path / "prior_run"
    checkpoint = prior / "portable_checkpoints" / "embedding_cache"
    cache = checkpoint / "embedding_cache"
    prepared = checkpoint / "prepared" / "modeling_cohort.parquet"
    historical = tmp_path / "prior_scratch" / "prepared" / "modeling_cohort.parquet"
    historical_manifest = historical.parent / "preparation_manifest.json"
    cache.mkdir(parents=True)
    prepared.parent.mkdir(parents=True)
    historical.parent.mkdir(parents=True)
    cohort_bytes = b"authenticated prepared cohort fixture"
    prepared.write_bytes(cohort_bytes)
    historical.write_bytes(cohort_bytes)
    historical_manifest.write_text("{}\n", encoding="utf-8")
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "production_provenance": {
                    "dataset": {"path": str(historical)}
                }
            }
        ),
        encoding="utf-8",
    )
    for name in ("chunk_embeddings.npy", "offsets.npy", "chunk_texts.jsonl"):
        (cache / name).write_bytes(f"fixture:{name}".encode("utf-8"))
    payloads = tuple(
        path.relative_to(checkpoint).as_posix()
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    )
    publish_portable_artifact(
        root=checkpoint,
        artifact_kind="embedding_cache",
        artifact_schema="test_cloud_embedding_cache_v1",
        compatibility=ArtifactCompatibility(
            dataset_identity="1" * 64,
            split_identity="2" * 64,
            row_order_identity="3" * 64,
            model_identities={"embedding": "4" * 64},
            prompt_identities={},
            configuration_identity="5" * 64,
            seed_identity="6" * 64,
            producer_code_identity="7" * 64,
            runtime_compatibility_class="test-runtime",
        ),
        upstream_artifact_ids=(),
        payload_paths=payloads,
        workflow_phase="embedding_cache",
        workflow_phase_result={
            "schema_version": "test_embedding_cache_phase_v1",
            "mode": "fresh_build",
            "cache_path": str(cache),
            "prepared_cohort_path": str(prepared),
            "terminal_files": [str(checkpoint / relative) for relative in payloads],
        },
    )

    assert resolver.resolve_import_inputs(prior) == (
        cache,
        historical,
        historical_manifest,
    )
    (cache / "offsets.npy").write_bytes(b"altered")
    with pytest.raises(ValueError, match="payload"):
        resolver.resolve_import_inputs(prior)
    (cache / "offsets.npy").write_bytes(b"fixture:offsets.npy")
    historical_manifest.unlink()
    with pytest.raises(FileNotFoundError):
        resolver.resolve_import_inputs(prior)
