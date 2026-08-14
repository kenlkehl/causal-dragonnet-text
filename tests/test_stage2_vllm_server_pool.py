from __future__ import annotations

import json
import os
import signal
from contextlib import contextmanager
from pathlib import Path

import pytest

import oci.inference.plain_handoff_stage2 as stage2_workflow
import oci.inference.vllm_server_pool as server_pool
from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2,
    PlainHandoffStage2Config,
    plain_stage2_config_from_mapping,
    run_plain_handoff_stage2,
)
from oci.inference.vllm_server_pool import (
    ManagedVLLMConfig,
    ManagedVLLMServerPool,
)
from oci.inference.research_all_evidence_workflow import (
    _raw_config_from_args,
    build_parser,
    compile_config,
)


def test_managed_gemma_defaults_and_command_are_text_only_and_thinking_enabled():
    config = plain_stage2_config_from_mapping(
        {
            "model": "google/gemma-4-31B-it",
            "vllm": {
                "server_count": 2,
                "gpus": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                "download_dir": "/models/cache",
                "extra_args": ["--gpu-memory-utilization", "0.85"],
            },
        },
        default_workers=8,
    )

    assert config is not None
    assert config.endpoint == ""
    assert config.enable_thinking is True
    assert config.vllm is not None
    assert config.vllm.reasoning_parser == "gemma4"
    assert config.vllm.language_model_only is True
    assert config.vllm.default_chat_template_kwargs == {"enable_thinking": True}
    assert config.vllm.gpu_groups() == (
        ("cuda:0", "cuda:1"),
        ("cuda:2", "cuda:3"),
    )

    command = server_pool._build_vllm_command(
        config.vllm,
        model=config.model,
        api_key="EMPTY",
        port=8010,
        tensor_parallel_size=2,
    )

    assert command[:5] == [
        server_pool.sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "google/gemma-4-31B-it",
    ]
    assert command[command.index("--reasoning-parser") + 1] == "gemma4"
    assert "--language-model-only" in command
    assert command[command.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":true}'
    )
    assert command[command.index("--download-dir") + 1] == "/models/cache"
    assert command[-2:] == ["--gpu-memory-utilization", "0.85"]


def test_managed_qwen_defaults_can_be_overridden():
    default_config = plain_stage2_config_from_mapping(
        {
            "model": "Qwen/Qwen3-32B",
            "vllm": {"server_count": 2, "gpus": "0,1"},
        },
        default_workers=4,
    )
    overridden = plain_stage2_config_from_mapping(
        {
            "model": "Qwen/Qwen3-32B",
            "enable_thinking": True,
            "vllm": {
                "server_count": 2,
                "gpus": [0, 1],
                "reasoning_parser": "custom-parser",
                "language_model_only": False,
                "default_chat_template_kwargs": {"enable_thinking": True},
            },
        },
        default_workers=4,
    )

    assert default_config is not None and default_config.vllm is not None
    assert default_config.vllm.reasoning_parser == "qwen3"
    assert default_config.vllm.language_model_only is True
    assert default_config.vllm.default_chat_template_kwargs is None
    assert default_config.enable_thinking is False
    assert overridden is not None and overridden.vllm is not None
    assert overridden.vllm.reasoning_parser == "custom-parser"
    assert overridden.vllm.language_model_only is False
    assert overridden.enable_thinking is True


@pytest.mark.parametrize(
    ("vllm", "message"),
    [
        (
            {"server_count": 3, "gpus": [0, 1, 2, 3]},
            "divide evenly",
        ),
        (
            {"server_count": 3, "gpus": [0, 1]},
            "cannot exceed",
        ),
        (
            {"server_count": 2, "gpus": [0, 0]},
            "duplicate",
        ),
        (
            {"server_count": 1, "gpus": [0], "extra_args": ["--port=9000"]},
            "cannot override",
        ),
    ],
)
def test_managed_vllm_rejects_unsafe_or_ambiguous_layouts(vllm, message):
    with pytest.raises(ValueError, match=message):
        plain_stage2_config_from_mapping(
            {"model": "Qwen/Qwen3-32B", "vllm": vllm},
            default_workers=4,
        )


def test_external_endpoint_and_managed_pool_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either stage2.endpoint or stage2.vllm"):
        plain_stage2_config_from_mapping(
            {
                "endpoint": "http://stage2.test/v1",
                "model": "Qwen/Qwen3-32B",
                "vllm": {"server_count": 1, "gpus": [0]},
            },
            default_workers=4,
        )


def test_managed_vllm_cli_options_compile_into_stage2_config(tmp_path):
    args = build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "cohort.parquet"),
            "--output-dir",
            str(tmp_path / "output"),
            "--stage2-only",
            "--stage2-model",
            "Qwen/Qwen3-32B",
            "--stage2-vllm-servers",
            "2",
            "--stage2-vllm-gpus",
            "cuda:0,cuda:1",
            "--stage2-vllm-base-port",
            "9100",
            "--stage2-vllm-download-dir",
            "/models/cache",
            "--stage2-vllm-reasoning-parser",
            "custom-qwen",
            "--no-stage2-vllm-language-model-only",
            "--stage2-vllm-default-chat-template-kwargs",
            '{"enable_thinking":true}',
            "--stage2-vllm-extra-arg=--max-model-len",
            "--stage2-vllm-extra-arg=65536",
        ]
    )
    raw, config_dir = _raw_config_from_args(args)
    config = compile_config(raw, config_dir=config_dir)

    assert config.mode == "stage2"
    assert config.stage2 is not None and config.stage2.vllm is not None
    assert config.stage2.vllm.server_count == 2
    assert config.stage2.vllm.gpus == ("cuda:0", "cuda:1")
    assert config.stage2.vllm.effective_ports() == (9100, 9101)
    assert config.stage2.vllm.download_dir == "/models/cache"
    assert config.stage2.vllm.reasoning_parser == "custom-qwen"
    assert config.stage2.vllm.language_model_only is False
    assert config.stage2.vllm.default_chat_template_kwargs == {"enable_thinking": True}
    assert config.stage2.vllm.extra_args == ("--max-model-len", "65536")


def test_visible_gpu_mapping_respects_parent_cuda_remapping():
    assert server_pool._visible_gpu_tokens(
        ("cuda:0", "cuda:2"),
        parent_cuda_visible_devices="5,GPU-abcdef,7",
    ) == ("5", "7")

    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        server_pool._visible_gpu_tokens(
            ("cuda:3",),
            parent_cuda_visible_devices="5,7",
        )
    with pytest.raises(ValueError, match="hides all GPUs"):
        server_pool._visible_gpu_tokens(
            ("cuda:0",),
            parent_cuda_visible_devices="",
        )


def test_stage2_round_robins_requests_and_transport_retries_across_endpoints(monkeypatch):
    called_endpoints: list[str] = []

    def fake_openai_completion(_messages, config):
        called_endpoints.append(config.endpoint)
        if len(called_endpoints) == 1:
            raise stage2_workflow._RetryableStage2ResponseError("temporary failure")
        return "{}"

    monkeypatch.setattr(stage2_workflow, "_openai_completion", fake_openai_completion)
    config = PlainHandoffStage2Config(
        endpoint="http://127.0.0.1:8010/v1",
        model="test-model",
        runtime_endpoints=(
            "http://127.0.0.1:8010/v1",
            "http://127.0.0.1:8011/v1",
            "http://127.0.0.1:8012/v1",
        ),
        transport_retry_backoff=0.0,
    )
    runner = PlainHandoffStage2(config=config, clinical_question="test")

    assert stage2_workflow._completion_with_transport_retries(
        [],
        runner.config,
        runner.completion,
    ) == "{}"
    for _ in range(5):
        assert runner.completion([], runner.config) == "{}"

    assert called_endpoints == [
        "http://127.0.0.1:8010/v1",
        "http://127.0.0.1:8011/v1",
        "http://127.0.0.1:8012/v1",
        "http://127.0.0.1:8010/v1",
        "http://127.0.0.1:8011/v1",
        "http://127.0.0.1:8012/v1",
        "http://127.0.0.1:8010/v1",
    ]


def test_run_wrapper_keeps_managed_servers_alive_for_stage2_and_cleans_up(
    tmp_path,
    monkeypatch,
):
    config = plain_stage2_config_from_mapping(
        {
            "model": "Qwen/Qwen3-32B",
            "vllm": {"server_count": 2, "gpus": [0, 1]},
        },
        default_workers=4,
    )
    assert config is not None
    lifecycle: list[str] = []
    captured = {}

    @contextmanager
    def fake_launch(**kwargs):
        lifecycle.append("started")
        assert kwargs["output_dir"] == tmp_path / "stage2" / "vllm_servers"
        try:
            yield ("http://127.0.0.1:8010/v1", "http://127.0.0.1:8011/v1")
        finally:
            lifecycle.append("stopped")

    def fake_run(self, **_kwargs):
        lifecycle.append("stage2")
        captured["config"] = self.config
        return {"ok": True}

    monkeypatch.setattr(stage2_workflow, "launch_managed_vllm_servers", fake_launch)
    monkeypatch.setattr(PlainHandoffStage2, "run", fake_run)

    result = run_plain_handoff_stage2(
        handoff_path=tmp_path / "handoff.jsonl",
        output_dir=tmp_path / "stage2",
        clinical_question="test",
        config=config,
    )

    assert result == {"ok": True}
    assert lifecycle == ["started", "stage2", "stopped"]
    assert captured["config"].runtime_endpoints == (
        "http://127.0.0.1:8010/v1",
        "http://127.0.0.1:8011/v1",
    )


def test_managed_pool_launches_one_process_per_gpu_and_writes_a_redacted_manifest(
    tmp_path,
    monkeypatch,
):
    popen_calls = []

    class FakeProcess:
        next_pid = 10_000

        def __init__(self, command, **kwargs):
            self.command = list(command)
            self.kwargs = kwargs
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.returncode = None
            popen_calls.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -signal.SIGTERM

        def kill(self):
            self.returncode = -signal.SIGKILL

        def wait(self, timeout=None):
            assert timeout is not None
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    monkeypatch.setenv("VLLM_PORT", "37973")
    monkeypatch.setattr(server_pool, "_assert_port_available", lambda *_args: None)
    monkeypatch.setattr(
        server_pool,
        "_server_model_ids",
        lambda *_args, **_kwargs: ["Qwen/Qwen3-32B"],
    )
    monkeypatch.setattr(server_pool.subprocess, "Popen", FakeProcess)

    def fake_signal(server, sig):
        if server.process.poll() is not None:
            return
        if sig == signal.SIGTERM:
            server.process.terminate()
        else:
            server.process.kill()

    monkeypatch.setattr(
        ManagedVLLMServerPool,
        "_signal_process_group",
        staticmethod(fake_signal),
    )
    config = ManagedVLLMConfig(
        server_count=2,
        gpus=("cuda:0", "cuda:1"),
        reasoning_parser="qwen3",
        language_model_only=True,
    )
    pool = ManagedVLLMServerPool(
        config=config,
        model="Qwen/Qwen3-32B",
        api_key="super-secret",
        output_dir=tmp_path / "servers",
    )

    assert pool.start() == (
        "http://127.0.0.1:8010/v1",
        "http://127.0.0.1:8011/v1",
    )
    assert [call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for call in popen_calls] == [
        "4",
        "6",
    ]
    assert [call.kwargs["env"]["VLLM_PORT"] for call in popen_calls] == [
        "20000",
        "20128",
    ]
    interpreter_bin = str(Path(server_pool.sys.executable).parent)
    assert all(
        call.kwargs["env"]["PATH"].split(os.pathsep)[0] == interpreter_bin
        for call in popen_calls
    )
    assert all("--tensor-parallel-size" in call.command for call in popen_calls)
    assert all(
        call.command[call.command.index("--tensor-parallel-size") + 1] == "1"
        for call in popen_calls
    )

    pool.stop()

    manifest_text = (tmp_path / "servers" / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["status"] == "stopped"
    assert [server["exit_code"] for server in manifest["servers"]] == [
        -signal.SIGTERM,
        -signal.SIGTERM,
    ]
    assert "super-secret" not in manifest_text
    assert "<redacted>" in manifest_text
    assert [server["vllm_internal_port_base"] for server in manifest["servers"]] == [
        20_000,
        20_128,
    ]
    assert all(call.kwargs["stdout"].closed for call in popen_calls)


def test_managed_pool_cleans_up_other_servers_when_one_exits_during_startup(
    tmp_path,
    monkeypatch,
):
    processes = []

    class FakeProcess:
        def __init__(self, _command, **kwargs):
            self.pid = 20_000 + len(processes)
            self.returncode = 17 if processes else None
            self.stdout_handle = kwargs["stdout"]
            processes.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -signal.SIGTERM

        def kill(self):
            self.returncode = -signal.SIGKILL

        def wait(self, timeout=None):
            assert timeout is not None
            return self.returncode

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(server_pool, "_assert_port_available", lambda *_args: None)
    monkeypatch.setattr(
        server_pool,
        "_server_model_ids",
        lambda *_args, **_kwargs: ["Qwen/Qwen3-32B"],
    )
    monkeypatch.setattr(server_pool.subprocess, "Popen", FakeProcess)

    def fake_startup_failure_signal(server, sig):
        if server.process.poll() is not None:
            return
        if sig == signal.SIGTERM:
            server.process.terminate()
        else:
            server.process.kill()

    monkeypatch.setattr(
        ManagedVLLMServerPool,
        "_signal_process_group",
        staticmethod(fake_startup_failure_signal),
    )
    pool = ManagedVLLMServerPool(
        config=ManagedVLLMConfig(
            server_count=2,
            gpus=("cuda:0", "cuda:1"),
            startup_poll_interval=0.01,
        ),
        model="Qwen/Qwen3-32B",
        api_key="EMPTY",
        output_dir=tmp_path / "servers",
    )

    with pytest.raises(RuntimeError, match="server 1 exited with code 17"):
        pool.start()

    assert [process.returncode for process in processes] == [-signal.SIGTERM, 17]
    assert all(process.stdout_handle.closed for process in processes)
    manifest = json.loads(
        (tmp_path / "servers" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert "server 1 exited with code 17" in manifest["error"]


def test_managed_pool_context_stops_before_propagating_termination_signal(
    tmp_path,
    monkeypatch,
):
    stops = []

    class FakePool:
        def __init__(self, **_kwargs):
            self.stopped = False

        def start(self):
            return ("http://127.0.0.1:8010/v1",)

        def stop(self, *, final_status="stopped"):
            if not self.stopped:
                stops.append(final_status)
                self.stopped = True

    monkeypatch.setattr(server_pool, "ManagedVLLMServerPool", FakePool)
    original_handler = signal.getsignal(signal.SIGTERM)
    context = server_pool.launch_managed_vllm_servers(
        config=ManagedVLLMConfig(server_count=1, gpus=("cuda:0",)),
        model="test-model",
        api_key="EMPTY",
        output_dir=tmp_path / "servers",
    )
    assert context.__enter__() == ("http://127.0.0.1:8010/v1",)
    installed_handler = signal.getsignal(signal.SIGTERM)
    assert callable(installed_handler)

    try:
        with pytest.raises(SystemExit) as exc_info:
            installed_handler(signal.SIGTERM, None)
        assert exc_info.value.code == 128 + signal.SIGTERM
    finally:
        context.__exit__(None, None, None)

    assert stops == ["interrupted"]
    assert signal.getsignal(signal.SIGTERM) is original_handler
