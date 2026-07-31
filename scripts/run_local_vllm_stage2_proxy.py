#!/usr/bin/env python3
"""Lazily start and supervise the local eight-GPU Stage 2 vLLM server.

The proxy binds before the production workflow starts but does not initialize
CUDA.  The first Stage 2 HTTP request arrives only after Stage 1 and its GPU
safety checks are complete; that request starts vLLM and waits for the pinned
model to become ready.  This keeps Stage 1 and the tensor-parallel server from
competing for the same eight GPUs.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import signal
import socket
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Mapping


PROXY_SCHEMA = "production_lazy_local_vllm_stage2_proxy_v1"
EXPECTED_ARCHITECTURE = "Gemma4ForConditionalGeneration"
EXPECTED_CONTEXT_TOKENS = 262_144
EXPECTED_QUANTIZATION = "NVFP4"
EXPECTED_QUANT_METHOD = "modelopt"
EXPECTED_TENSOR_PARALLEL_SIZE = 8
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _regular_executable(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    state = resolved.stat()
    if not stat.S_ISREG(state.st_mode) or not os.access(resolved, os.X_OK):
        raise ValueError(f"vLLM command is not one executable regular file: {resolved}")
    return resolved


def _real_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not one directory: {resolved}")
    return resolved


def _validate_model_tree(path: Path) -> None:
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        legacy_quant = json.loads(
            (path / "hf_quant_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("local vLLM model metadata is unreadable") from exc
    text_config = config.get("text_config")
    quantization = config.get("quantization_config")
    if (
        config.get("architectures") != [EXPECTED_ARCHITECTURE]
        or config.get("model_type") != "gemma4"
        or not isinstance(text_config, Mapping)
        or text_config.get("max_position_embeddings") != EXPECTED_CONTEXT_TOKENS
        or not isinstance(quantization, Mapping)
        or quantization.get("quant_method") != EXPECTED_QUANT_METHOD
        or quantization.get("quant_algo") != EXPECTED_QUANTIZATION
        or legacy_quant.get("producer", {}).get("name") != EXPECTED_QUANT_METHOD
        or legacy_quant.get("quantization", {}).get("quant_algo")
        != EXPECTED_QUANTIZATION
    ):
        raise ValueError(
            "local vLLM model is not the expected 256K Gemma 4 ModelOpt NVFP4 checkpoint"
        )


@dataclass(frozen=True)
class ProxyConfiguration:
    listen_host: str
    listen_port: int
    upstream_host: str
    upstream_port: int
    vllm_command: Path
    model_dir: Path
    served_model_name: str
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    startup_timeout_seconds: float
    request_timeout_seconds: float
    log_path: Path
    status_path: Path

    def __post_init__(self) -> None:
        if self.listen_host != "127.0.0.1" or self.upstream_host != "127.0.0.1":
            raise ValueError("the Stage 2 proxy and vLLM upstream must be loopback-only")
        for name in ("listen_port", "upstream_port"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1024 <= value <= 65535:
                raise ValueError(f"{name} must be an unprivileged TCP port")
        if self.listen_port == self.upstream_port:
            raise ValueError("proxy and vLLM upstream ports must differ")
        if self.tensor_parallel_size != EXPECTED_TENSOR_PARALLEL_SIZE:
            raise ValueError("Stage 2 vLLM tensor parallelism must use all eight GPUs")
        if self.max_model_len != EXPECTED_CONTEXT_TOKENS:
            raise ValueError("Stage 2 vLLM must expose the complete 256K context window")
        if (
            isinstance(self.max_num_seqs, bool)
            or not isinstance(self.max_num_seqs, int)
            or self.max_num_seqs < 1
        ):
            raise ValueError("max_num_seqs must be positive")
        if (
            isinstance(self.gpu_memory_utilization, bool)
            or not math.isfinite(self.gpu_memory_utilization)
            or not 0.0 < self.gpu_memory_utilization <= 0.95
        ):
            raise ValueError("gpu_memory_utilization must be in (0, 0.95]")
        for name in ("startup_timeout_seconds", "request_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not isinstance(self.served_model_name, str)
            or not self.served_model_name
            or self.served_model_name != self.served_model_name.strip()
        ):
            raise ValueError("served_model_name must be one nonempty exact alias")

        command = _regular_executable(self.vllm_command)
        model = _real_directory(self.model_dir, label="Stage 2 vLLM model")
        _validate_model_tree(model)
        object.__setattr__(self, "vllm_command", command)
        object.__setattr__(self, "model_dir", model)
        for target in (self.log_path, self.status_path):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and (target.is_symlink() or not target.is_file()):
                raise ValueError(f"operational target is not a regular file: {target}")

    def command(self) -> tuple[str, ...]:
        return (
            str(self.vllm_command),
            "serve",
            str(self.model_dir),
            "--served-model-name",
            self.served_model_name,
            "--tokenizer",
            str(self.model_dir),
            "--quantization",
            EXPECTED_QUANT_METHOD,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--distributed-executor-backend",
            "mp",
            "--model-impl",
            "vllm",
            "--host",
            self.upstream_host,
            "--port",
            str(self.upstream_port),
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            format(self.gpu_memory_utilization, ".6g"),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--generation-config",
            "vllm",
            "--reasoning-parser",
            "gemma4",
            "--seed",
            "42",
            "--async-scheduling",
        )

    def public_identity(self) -> dict[str, object]:
        return {
            "schema_version": PROXY_SCHEMA,
            "listen_endpoint": f"http://{self.listen_host}:{self.listen_port}/v1",
            "upstream_endpoint": f"http://{self.upstream_host}:{self.upstream_port}/v1",
            "served_model_name": self.served_model_name,
            "model_dir": str(self.model_dir),
            "quantization": EXPECTED_QUANT_METHOD,
            "quantization_algorithm": EXPECTED_QUANTIZATION,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "startup_policy": "lazy_on_first_stage2_request_after_stage1_v1",
            "vllm_command": list(self.command()),
        }


class VllmSupervisor:
    def __init__(self, configuration: ProxyConfiguration) -> None:
        self.configuration = configuration
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._log_stream = None
        self._ready = False
        self._failure: str | None = None
        self._shutdown_requested = threading.Event()
        self._write_status("proxy_ready_vllm_not_started")

    @property
    def process(self) -> subprocess.Popen[bytes] | None:
        return self._process

    def _write_status(self, state: str, *, detail: str | None = None) -> None:
        body: dict[str, object] = {
            **self.configuration.public_identity(),
            "state": state,
            "proxy_pid": os.getpid(),
            "vllm_pid": None if self._process is None else self._process.pid,
            "vllm_process_group": None if self._process is None else self._process.pid,
            "updated_unix_seconds": time.time(),
        }
        if detail is not None:
            body["detail"] = detail
        temporary = self.configuration.status_path.with_name(
            f".{self.configuration.status_path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        temporary.write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.configuration.status_path)

    def _upstream_models(self) -> set[str] | None:
        health_url = (
            f"http://{self.configuration.upstream_host}:"
            f"{self.configuration.upstream_port}/health"
        )
        models_url = (
            f"http://{self.configuration.upstream_host}:"
            f"{self.configuration.upstream_port}/v1/models"
        )
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                if response.status != 200:
                    return None
            with urllib.request.urlopen(models_url, timeout=2.0) as response:
                if response.status != 200:
                    return None
                payload = json.loads(response.read().decode("utf-8"))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ):
            return None
        rows = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            return None
        return {
            str(row["id"])
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("id"), str)
        }

    def _failure_tail(self) -> str:
        try:
            rows = self.configuration.log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return "vLLM log unavailable"
        return "\n".join(rows[-40:])

    def _upstream_port_is_open(self) -> bool:
        try:
            with socket.create_connection(
                (
                    self.configuration.upstream_host,
                    self.configuration.upstream_port,
                ),
                timeout=0.5,
            ):
                return True
        except OSError:
            return False

    def ensure_ready(self) -> None:
        with self._lock:
            if self._failure is not None:
                raise RuntimeError(self._failure)
            if self._ready:
                if self._process is None or self._process.poll() is not None:
                    self._failure = "vLLM exited after becoming ready"
                    self._write_status("vllm_failed", detail=self._failure)
                    raise RuntimeError(self._failure)
                return
            if self._process is not None:
                raise RuntimeError("vLLM startup state is inconsistent")
            if self._upstream_models() is not None or self._upstream_port_is_open():
                raise RuntimeError("vLLM upstream port was occupied before owned startup")

            self._log_stream = self.configuration.log_path.open("ab", buffering=0)
            marker = (
                f"\n[{time.time():.6f}] starting owned Stage 2 vLLM: "
                + json.dumps(list(self.configuration.command()))
                + "\n"
            ).encode("utf-8")
            self._log_stream.write(marker)
            self._process = subprocess.Popen(
                self.configuration.command(),
                stdin=subprocess.DEVNULL,
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            self._write_status("vllm_starting")
            deadline = time.monotonic() + self.configuration.startup_timeout_seconds
            while time.monotonic() < deadline:
                if self._shutdown_requested.is_set():
                    raise RuntimeError("vLLM startup interrupted by launcher shutdown")
                return_code = self._process.poll()
                if return_code is not None:
                    self._failure = (
                        f"vLLM exited during startup with status {return_code}\n"
                        f"{self._failure_tail()}"
                    )
                    self._write_status("vllm_failed", detail=self._failure)
                    raise RuntimeError(self._failure)
                models = self._upstream_models()
                if models is not None:
                    if self.configuration.served_model_name not in models:
                        self._failure = (
                            "vLLM model registry does not expose the exact pinned alias; "
                            f"observed={sorted(models)}"
                        )
                        self._write_status("vllm_failed", detail=self._failure)
                        raise RuntimeError(self._failure)
                    self._ready = True
                    self._write_status("vllm_ready")
                    return
                time.sleep(2.0)
            self._failure = (
                "vLLM did not become ready before the configured startup timeout\n"
                f"{self._failure_tail()}"
            )
            self._write_status("vllm_failed", detail=self._failure)
            raise RuntimeError(self._failure)

    def shutdown(self) -> None:
        self._shutdown_requested.set()
        with self._lock:
            process = self._process
            if process is None:
                self._write_status("proxy_stopped_without_vllm_start")
                return
            if process.poll() is None:
                try:
                    process_group = os.getpgid(process.pid)
                except ProcessLookupError:
                    process_group = None
                if process_group != process.pid:
                    self._write_status(
                        "vllm_shutdown_identity_rejected",
                        detail="owned vLLM process-group identity changed",
                    )
                    return
                os.killpg(process_group, signal.SIGTERM)
                try:
                    process.wait(timeout=60.0)
                except subprocess.TimeoutExpired:
                    self._write_status(
                        "vllm_sigterm_pending",
                        detail="no SIGKILL was sent",
                    )
                    return
            self._write_status(
                "vllm_stopped",
                detail=f"exit_status={process.returncode}",
            )
            if self._log_stream is not None:
                self._log_stream.close()
                self._log_stream = None


class LazyProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, configuration: ProxyConfiguration, supervisor: VllmSupervisor) -> None:
        self.configuration = configuration
        self.supervisor = supervisor
        super().__init__(
            (configuration.listen_host, configuration.listen_port),
            LazyProxyHandler,
        )


class LazyProxyHandler(BaseHTTPRequestHandler):
    server: LazyProxyServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        print(
            f"[Stage 2 proxy] {self.client_address[0]} " + format % args,
            flush=True,
        )

    def _json_error(self, status: int, message: str) -> None:
        payload = json.dumps(
            {"error": {"message": message, "type": "local_vllm_proxy_error"}},
            sort_keys=True,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def _proxy_health(self) -> None:
        process = self.server.supervisor.process
        payload = json.dumps(
            {
                "schema_version": PROXY_SCHEMA,
                "status": "ready",
                "vllm_started": process is not None,
                "vllm_running": process is not None and process.poll() is None,
            },
            sort_keys=True,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def _forward(self) -> None:
        if self.path == "/proxy-health":
            self._proxy_health()
            return
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            body = None
        else:
            try:
                length = int(raw_length)
            except ValueError:
                self._json_error(400, "invalid Content-Length")
                return
            if length < 0 or length > 64 * 1024 * 1024:
                self._json_error(413, "request body exceeds the proxy limit")
                return
            body = self.rfile.read(length)

        try:
            self.server.supervisor.ensure_ready()
        except Exception as exc:
            self._json_error(503, str(exc))
            return

        request_headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS | {"host", "content-length"}
        }
        connection = http.client.HTTPConnection(
            self.server.configuration.upstream_host,
            self.server.configuration.upstream_port,
            timeout=self.server.configuration.request_timeout_seconds,
        )
        try:
            connection.request(
                self.command,
                self.path,
                body=body,
                headers=request_headers,
            )
            response = connection.getresponse()
            response_body = response.read()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_BY_HOP_HEADERS | {"content-length"}:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(response_body)
        except (OSError, http.client.HTTPException) as exc:
            self._json_error(502, f"owned vLLM transport failed: {exc}")
        finally:
            connection.close()

    do_GET = _forward
    do_POST = _forward


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, required=True)
    parser.add_argument("--vllm-command", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=262_144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--startup-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def _configuration(args: argparse.Namespace) -> ProxyConfiguration:
    return ProxyConfiguration(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream_host=args.upstream_host,
        upstream_port=args.upstream_port,
        vllm_command=args.vllm_command,
        model_dir=args.model_dir,
        served_model_name=args.served_model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        startup_timeout_seconds=args.startup_timeout_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        log_path=args.log_path,
        status_path=args.status_path,
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configuration = _configuration(args)
    if args.check_only:
        print(json.dumps(configuration.public_identity(), indent=2, sort_keys=True))
        return 0

    supervisor = VllmSupervisor(configuration)
    server = LazyProxyServer(configuration, supervisor)

    def stop(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(
        f"[Stage 2 proxy] listening on "
        f"http://{configuration.listen_host}:{configuration.listen_port}; "
        "vLLM will start lazily on the first Stage 2 request",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        supervisor.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
