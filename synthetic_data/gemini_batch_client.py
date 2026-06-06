"""Gemini batch inference utilities for synthetic data generation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import json
import logging
import re
import time


logger = logging.getLogger(__name__)

REQUEST_ID_RE = re.compile(r"REQUEST_ID:\s*([A-Za-z0-9_.:-]+)")
TERMINAL_JOB_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
    "JOB_STATE_EXPIRED",
}
FAILED_JOB_STATES = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
    "JOB_STATE_EXPIRED",
}


@dataclass
class GeminiBatchConfig:
    """Configuration for Gemini batch inference on Agent Platform / Vertex AI."""

    project: str
    staging_uri: str
    location: str = "us-central1"
    model_name: str = "gemini-2.5-flash-lite"
    batch_max_requests: int = 100_000
    batch_max_input_bytes: int = 800_000_000
    poll_interval_seconds: int = 30
    submit_only: bool = False

    def validate(self) -> None:
        if not self.project:
            raise ValueError("Gemini batch requires a GCP project")
        if not self.staging_uri.startswith("gs://"):
            raise ValueError("Gemini batch staging_uri must be a gs:// URI")
        if self.batch_max_requests < 1:
            raise ValueError("batch_max_requests must be at least 1")
        if self.batch_max_input_bytes < 1:
            raise ValueError("batch_max_input_bytes must be at least 1")


def build_gemini_batch_request(
    prompt: str,
    *,
    request_id: str,
    system_prompt: Optional[str],
    temperature: float,
    max_output_tokens: int,
) -> Dict[str, Any]:
    """Build one Cloud Storage JSONL request for Gemini batch inference."""
    marked_prompt = f"REQUEST_ID: {request_id}\n\n{prompt}"
    request: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": marked_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if system_prompt:
        request["systemInstruction"] = {
            "parts": [{"text": system_prompt}],
        }
    return {"request": request}


def extract_request_id(output_record: Dict[str, Any]) -> Optional[str]:
    """Extract the synthetic request id from an echoed Gemini batch request."""
    request = output_record.get("request") or {}
    for content in request.get("contents", []) or []:
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if not text:
                continue
            match = REQUEST_ID_RE.search(text)
            if match:
                return match.group(1)
    return None


def extract_response_text(output_record: Dict[str, Any]) -> str:
    """Extract generated text from a Gemini batch output record."""
    response = output_record.get("response") or {}
    candidates = response.get("candidates") or []
    if not candidates:
        return ""

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    return "".join(part.get("text", "") for part in parts).strip()


def iter_jsonl_records(paths: Iterable[Path]) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from one or more JSONL files, skipping empty lines."""
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSONL line %s:%d: %s", path, line_number, exc)


def manifest_matches_shards(manifest: Dict[str, Any], shards: List[Dict[str, Any]]) -> bool:
    """Return True if a saved manifest describes the request shards just written."""
    previous_shards = manifest.get("shards") or []
    if len(previous_shards) != len(shards):
        return False

    for previous, current in zip(previous_shards, shards):
        for key in ("stage", "shard_index", "request_count", "input_bytes"):
            if previous.get(key) != current.get(key):
                return False

        previous_name = Path(str(previous.get("local_input_path", ""))).name
        current_name = Path(str(current.get("local_input_path", ""))).name
        if previous_name != current_name:
            return False

    return True


class GeminiRequestShardWriter:
    """Write Gemini batch request JSONL shards under request-count and byte limits."""

    def __init__(
        self,
        *,
        stage: str,
        input_dir: Path,
        config: GeminiBatchConfig,
        system_prompt: Optional[str],
        temperature: float,
        max_output_tokens: int,
    ):
        self.stage = stage
        self.input_dir = input_dir
        self.config = config
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.input_dir.mkdir(parents=True, exist_ok=True)

        self._file = None
        self._shard_index = -1
        self._current_count = 0
        self._current_bytes = 0
        self._current_path: Optional[Path] = None
        self.shards: List[Dict[str, Any]] = []

    def _start_shard(self) -> None:
        self._shard_index += 1
        self._current_count = 0
        self._current_bytes = 0
        self._current_path = self.input_dir / f"{self.stage}-{self._shard_index:05d}.jsonl"
        self._file = open(self._current_path, "w", encoding="utf-8")

    def _close_current_shard(self) -> None:
        if self._file is None or self._current_path is None:
            return
        self._file.close()
        self.shards.append(
            {
                "stage": self.stage,
                "shard_index": self._shard_index,
                "local_input_path": str(self._current_path),
                "request_count": self._current_count,
                "input_bytes": self._current_bytes,
            }
        )
        self._file = None
        self._current_path = None

    def add(self, request_id: str, prompt: str) -> None:
        record = build_gemini_batch_request(
            prompt,
            request_id=request_id,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        encoded_len = len(encoded) + 1
        if encoded_len > self.config.batch_max_input_bytes:
            raise ValueError(
                f"Single Gemini batch request {request_id} is {encoded_len} bytes, "
                f"larger than shard limit {self.config.batch_max_input_bytes}"
            )

        if self._file is None:
            self._start_shard()
        elif (
            self._current_count >= self.config.batch_max_requests
            or self._current_bytes + encoded_len > self.config.batch_max_input_bytes
        ):
            self._close_current_shard()
            self._start_shard()

        assert self._file is not None
        self._file.write(encoded.decode("utf-8"))
        self._file.write("\n")
        self._current_count += 1
        self._current_bytes += encoded_len

    def close(self) -> List[Dict[str, Any]]:
        self._close_current_shard()
        return self.shards

    def __enter__(self) -> "GeminiRequestShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class GeminiBatchClient:
    """Small wrapper around Gemini batch jobs and Cloud Storage JSONL files."""

    def __init__(self, config: GeminiBatchConfig):
        config.validate()
        self.config = config
        self._genai_client = None
        self._storage_client = None

    @staticmethod
    def parse_gcs_uri(uri: str) -> Tuple[str, str]:
        if not uri.startswith("gs://"):
            raise ValueError(f"Expected gs:// URI, got {uri!r}")
        path = uri[5:]
        bucket, _, prefix = path.partition("/")
        if not bucket:
            raise ValueError(f"Invalid GCS URI: {uri!r}")
        return bucket, prefix.rstrip("/")

    @staticmethod
    def _join_gcs(prefix: str, *parts: str) -> str:
        return "/".join([prefix.rstrip("/"), *(part.strip("/") for part in parts if part)])

    @staticmethod
    def _model_for_sdk(model_name: str) -> str:
        return model_name.rstrip("/").split("/")[-1]

    @staticmethod
    def _state_name(state: Any) -> str:
        value = getattr(state, "name", None) or str(state)
        return value.split(".")[-1]

    def _get_genai_client(self):
        if self._genai_client is not None:
            return self._genai_client

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "Gemini batch generation requires google-genai. "
                "Install with: pip install '.[gemini]'"
            ) from exc

        http_options = types.HttpOptions(api_version="v1")
        try:
            self._genai_client = genai.Client(
                enterprise=True,
                project=self.config.project,
                location=self.config.location,
                http_options=http_options,
            )
        except TypeError:
            self._genai_client = genai.Client(
                vertexai=True,
                project=self.config.project,
                location=self.config.location,
                http_options=http_options,
            )
        return self._genai_client

    def _get_storage_client(self):
        if self._storage_client is not None:
            return self._storage_client
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ImportError(
                "Gemini batch generation requires google-cloud-storage. "
                "Install with: pip install '.[gemini]'"
            ) from exc
        self._storage_client = storage.Client(project=self.config.project)
        return self._storage_client

    def open_request_writer(
        self,
        *,
        stage: str,
        input_dir: Path,
        system_prompt: Optional[str],
        temperature: float,
        max_output_tokens: int,
    ) -> GeminiRequestShardWriter:
        return GeminiRequestShardWriter(
            stage=stage,
            input_dir=input_dir,
            config=self.config,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def write_request_shards(
        self,
        *,
        stage: str,
        input_dir: Path,
        requests: Iterable[Tuple[str, str]],
        system_prompt: Optional[str],
        temperature: float,
        max_output_tokens: int,
    ) -> List[Dict[str, Any]]:
        with self.open_request_writer(
            stage=stage,
            input_dir=input_dir,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ) as writer:
            for request_id, prompt in requests:
                writer.add(request_id, prompt)
            return writer.close()

    def upload_file(self, local_path: Path, gcs_uri: str) -> None:
        storage_client = self._get_storage_client()
        bucket_name, blob_name = self.parse_gcs_uri(gcs_uri)
        bucket = storage_client.bucket(bucket_name)
        logger.info("Uploading %s to %s", local_path, gcs_uri)
        bucket.blob(blob_name).upload_from_filename(str(local_path))

    def download_prefix(self, gcs_uri_prefix: str, local_dir: Path) -> List[Path]:
        storage_client = self._get_storage_client()
        bucket_name, prefix = self.parse_gcs_uri(gcs_uri_prefix)
        bucket = storage_client.bucket(bucket_name)
        local_dir.mkdir(parents=True, exist_ok=True)

        downloaded: List[Path] = []
        for blob in storage_client.list_blobs(bucket, prefix=prefix):
            if blob.name.endswith("/") or blob.size == 0:
                continue
            relative = blob.name[len(prefix):].lstrip("/") if prefix else blob.name
            local_path = local_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading gs://%s/%s to %s", bucket_name, blob.name, local_path)
            blob.download_to_filename(str(local_path))
            downloaded.append(local_path)
        return downloaded

    def submit_job(self, *, input_uri: str, output_uri_prefix: str, display_name: str):
        client = self._get_genai_client()
        try:
            from google.genai import types
        except ImportError as exc:
            raise ImportError("Gemini batch generation requires google-genai") from exc

        try:
            job_config = types.CreateBatchJobConfig(
                dest=output_uri_prefix,
                display_name=display_name,
            )
        except TypeError:
            job_config = types.CreateBatchJobConfig(dest=output_uri_prefix)

        logger.info("Submitting Gemini batch job %s with input %s", display_name, input_uri)
        return client.batches.create(
            model=self._model_for_sdk(self.config.model_name),
            src=input_uri,
            config=job_config,
        )

    def wait_for_job(self, job_name: str):
        client = self._get_genai_client()
        job = client.batches.get(name=job_name)
        while self._state_name(getattr(job, "state", "")) not in TERMINAL_JOB_STATES:
            logger.info("Gemini batch job %s state: %s", job_name, getattr(job, "state", ""))
            time.sleep(self.config.poll_interval_seconds)
            job = client.batches.get(name=job_name)

        state = self._state_name(getattr(job, "state", ""))
        logger.info("Gemini batch job %s terminal state: %s", job_name, state)
        if state in FAILED_JOB_STATES:
            error = getattr(job, "error", None)
            raise RuntimeError(f"Gemini batch job {job_name} ended in {state}: {error}")
        return job

    def run_shards(
        self,
        *,
        stage: str,
        shards: List[Dict[str, Any]],
        stage_dir: Path,
        display_name_prefix: str,
    ) -> Dict[str, Any]:
        """Upload, submit, optionally wait, and download output for request shards."""
        stage_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = stage_dir / "manifest.json"
        manifest: Dict[str, Any] = {
            "config": asdict(self.config),
            "stage": stage,
            "remote_stage": stage,
            "shards": shards,
        }
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    previous = json.load(f)
                if previous.get("stage") == stage and manifest_matches_shards(previous, shards):
                    manifest = previous
                    shards = manifest["shards"]
                    logger.info("Loaded existing Gemini stage manifest: %s", manifest_path)
                elif previous.get("stage") == stage:
                    logger.info(
                        "Ignoring stale Gemini stage manifest with mismatched shards: %s",
                        manifest_path,
                    )
                    manifest["remote_stage"] = f"{stage}-repair-{int(time.time())}"
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load existing Gemini manifest %s: %s", manifest_path, exc)

        staging_root = self.config.staging_uri.rstrip("/")
        remote_stage = manifest.get("remote_stage", stage)
        for shard in shards:
            shard_index = shard["shard_index"]
            local_input_path = Path(shard["local_input_path"])
            input_uri = shard.get("gcs_input_uri") or self._join_gcs(
                staging_root, "inputs", remote_stage, local_input_path.name
            )
            output_uri_prefix = shard.get("gcs_output_uri_prefix") or self._join_gcs(
                staging_root, "outputs", remote_stage, f"shard-{shard_index:05d}"
            )
            shard["gcs_input_uri"] = input_uri
            shard["gcs_output_uri_prefix"] = output_uri_prefix

            if not shard.get("uploaded"):
                self.upload_file(local_input_path, input_uri)
                shard["uploaded"] = True
                self._write_manifest(manifest_path, manifest)

            if not shard.get("job_name"):
                display_name = f"{display_name_prefix}-{stage}-{shard_index:05d}"
                job = self.submit_job(
                    input_uri=input_uri,
                    output_uri_prefix=output_uri_prefix,
                    display_name=display_name,
                )
                shard["job_name"] = getattr(job, "name", None)
                shard["job_state"] = self._state_name(getattr(job, "state", ""))
                self._write_manifest(manifest_path, manifest)

        if self.config.submit_only:
            logger.info("Gemini submit-only mode enabled; not waiting for %s jobs", stage)
            self._write_manifest(manifest_path, manifest)
            return manifest

        for shard in shards:
            if shard.get("job_state") != "JOB_STATE_SUCCEEDED":
                job = self.wait_for_job(shard["job_name"])
                shard["job_state"] = self._state_name(getattr(job, "state", ""))
                self._write_manifest(manifest_path, manifest)

            output_dir = stage_dir / "outputs" / f"shard-{shard['shard_index']:05d}"
            local_output_paths = [
                path
                for path in shard.get("local_output_paths", []) or []
                if Path(path).exists()
            ]
            if len(local_output_paths) != len(shard.get("local_output_paths", []) or []):
                shard["local_output_paths"] = local_output_paths

            if not shard.get("local_output_paths"):
                downloaded = self.download_prefix(shard["gcs_output_uri_prefix"], output_dir)
                shard["local_output_paths"] = [str(path) for path in downloaded]
                self._write_manifest(manifest_path, manifest)

        self._write_manifest(manifest_path, manifest)
        return manifest

    @staticmethod
    def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
