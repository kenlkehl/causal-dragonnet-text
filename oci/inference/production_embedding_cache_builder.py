"""Offline, atomic producer for an arbitrary cohort's frozen embedding cache.

The Stage 1 production wrapper consumes a four-file, row-bound cache through
``SpentOnlyFrozenChunkEmbeddingCache``.  This module is the deliberately small
construction boundary for that cache: it accepts only a local model tree and a
local Parquet cohort, builds under a fresh temporary sibling, validates every
row against the production reader, and publishes by one directory rename.

No existing cache is resumed or repaired.  A failed build leaves neither a
partial target nor a reusable temporary directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import socket
import stat
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from oci.models.concept_embedding_utils import chunk_text_words

from .review_spent_evidence_provider import SpentOnlyFrozenChunkEmbeddingCache

PRODUCTION_EMBEDDING_CACHE_BUILDER_VERSION = (
    "production_arbitrary_cohort_embedding_cache_builder_v2"
)
PRODUCTION_EMBEDDING_CACHE_METADATA_SCHEMA = (
    "production_arbitrary_cohort_embedding_cache_metadata_v2"
)
PRODUCTION_EMBEDDING_CACHE_PROVENANCE_SCHEMA = (
    "production_arbitrary_cohort_embedding_cache_provenance_v2"
)
PRODUCTION_EMBEDDING_CACHE_RESULT_SCHEMA = "production_arbitrary_cohort_embedding_cache_result_v2"

_CACHE_FILES = frozenset(
    {
        "metadata.json",
        "chunk_embeddings.npy",
        "offsets.npy",
        "chunk_texts.jsonl",
    }
)
_COMPANION_FILES = (
    "chunk_embeddings.npy",
    "offsets.npy",
    "chunk_texts.jsonl",
)
_CHUNK_CONFIG_FIELDS = frozenset(
    {
        "chunk_size_words",
        "chunk_overlap_words",
        "max_chunks",
        "chunk_selection",
        "normalize_embeddings",
        "max_seq_length",
    }
)
_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "sentence_model_name",
        "hidden_size",
        "num_samples",
        "total_chunks",
        "chunk_counts",
        "chunk_size_words",
        "chunk_overlap_words",
        "max_chunks",
        "chunk_selection",
        "normalize_embeddings",
        "max_seq_length",
        "effective_max_seq_length",
        "chunking_mode",
        "actual_max_len",
        "uncapped_total_chunks",
        "uncapped_chunk_counts_sha256",
        "chunk_cap_nonbinding",
        "semantic_truncation_allowed",
        "max_observed_token_count",
        "ordered_token_counts_sha256",
        "tokenizer_truncation_allowed",
        "storage_format",
        "dtype",
        "production_provenance",
        "production_provenance_sha256",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "builder_version",
        "builder_code_sha256",
        "dataset",
        "sentence_model_name",
        "local_model",
        "chunk_configuration",
        "chunk_configuration_sha256",
        "cache_configuration_sha256",
        "encoder_execution",
        "companion_cache_files",
        "uncapped_total_chunks",
        "uncapped_chunk_counts_sha256",
        "chunk_cap_nonbinding",
        "semantic_truncation_allowed",
        "max_observed_token_count",
        "ordered_token_counts_sha256",
        "tokenizer_truncation_allowed",
        "atomic_publication",
        "partial_cache_reuse_allowed",
        "network_access_allowed",
        "symlinks_allowed",
        "executable_artifacts_allowed",
    }
)
_DATASET_PROVENANCE_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "text_column",
        "row_count",
        "ordered_text_sha256",
    }
)
_MODEL_PROVENANCE_FIELDS = frozenset(
    {"path", "tree_sha256", "file_count", "directory_count", "total_file_bytes"}
)
_ENCODER_EXECUTION_FIELDS = frozenset(
    {
        "device",
        "batch_size",
        "local_files_only",
        "trust_remote_code",
        "offline_environment",
        "socket_access_blocked",
    }
)
_FILE_REGISTRATION_FIELDS = frozenset({"sha256", "size_bytes"})
_PROVIDER_CACHE_HASH_FIELDS = {
    "metadata.json": "metadata_sha256",
    "chunk_embeddings.npy": "embeddings_sha256",
    "offsets.npy": "offsets_sha256",
    "chunk_texts.jsonl": "chunk_texts_sha256",
}
_PROVIDER_IDENTITY_FIELDS = frozenset(
    {
        "provider",
        *_PROVIDER_CACHE_HASH_FIELDS.values(),
        "row_count",
        "chunk_count",
        "cache_snapshot_authentication",
        "chunk_text_storage",
        "embeddings_path_backed",
        "private_snapshot_embedding_mmap",
        "future_row_text_decoded",
        "novel_text_encoding_allowed",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "builder_version",
        "builder_code_sha256",
        "cache_path",
        "production_provenance_sha256",
        "dataset_sha256",
        "ordered_text_sha256",
        "sentence_model_name",
        "local_model_tree_sha256",
        "chunk_configuration_sha256",
        "cache_configuration_sha256",
        "row_count",
        "chunk_count",
        "hidden_size",
        "cache_files",
        "provider_identity",
        "atomic_publication",
        "offline_build",
    }
)
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
_BANNED_EXECUTABLE_SUFFIXES = frozenset(
    {
        ".bat",
        ".bin",
        ".cmd",
        ".com",
        ".dll",
        ".dylib",
        ".exe",
        ".jar",
        ".joblib",
        ".pkl",
        ".pickle",
        ".ps1",
        ".pt",
        ".pth",
        ".py",
        ".pyc",
        ".pyo",
        ".sh",
        ".so",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OFFLINE_LOCK = threading.RLock()
_TOKEN_LENGTH_AUDIT_BATCH_SIZE = 128


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validated_sentence_model_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(
            "sentence_model_name must be a non-empty logical model identifier "
            "without surrounding whitespace or control characters"
        )
    return value


def _cache_configuration_sha256(
    *, sentence_model_name: str, chunk_configuration: Mapping[str, Any]
) -> str:
    return _sha256_json(
        {
            "schema_version": "production_embedding_cache_configuration_identity_v1",
            "sentence_model_name": sentence_model_name,
            "chunk_configuration": copy.deepcopy(dict(chunk_configuration)),
        }
    )


def _require_sha256(value: Any, *, field_name: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return text


def _builder_code_sha256() -> str:
    return _stable_file_snapshot(
        Path(__file__).resolve(strict=True),
        label="production embedding cache builder module",
    ).sha256


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _real_directory_signature(path: Path, *, label: str) -> tuple[int, int, int, int, int, int]:
    try:
        state = os.lstat(path)
    except OSError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise ValueError(f"{label} must be one real directory")
    return _stat_signature(state)


@dataclass(frozen=True)
class _FileSnapshot:
    sha256: str
    size_bytes: int
    signature: tuple[int, int, int, int, int, int]
    leading_bytes: bytes = field(repr=False)


def _stable_file_snapshot(path: Path, *, label: str) -> _FileSnapshot:
    source = Path(path)
    try:
        before_path = os.lstat(source)
    except OSError as exc:
        raise FileNotFoundError(f"{label} does not exist: {source}") from exc
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"{label} must be one real regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError(f"{label} could not be opened without following links") from exc
    digest = hashlib.sha256()
    leading = b""
    try:
        before_fd = os.fstat(descriptor)
        if _stat_signature(before_fd) != _stat_signature(before_path):
            raise RuntimeError(f"{label} changed while it was being opened")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            if len(leading) < 16:
                leading += block[: 16 - len(leading)]
            digest.update(block)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(source)
    signature = _stat_signature(before_path)
    if (
        _stat_signature(before_fd) != signature
        or _stat_signature(after_fd) != signature
        or _stat_signature(after_path) != signature
    ):
        raise RuntimeError(f"{label} changed while it was being authenticated")
    return _FileSnapshot(
        sha256=digest.hexdigest(),
        size_bytes=int(before_path.st_size),
        signature=signature,
        leading_bytes=leading,
    )


def _reject_executable_file(path: Path, snapshot: _FileSnapshot, *, label: str) -> None:
    mode = int(snapshot.signature[2])
    suffix = path.suffix.casefold()
    leading = snapshot.leading_bytes
    if (
        mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        or suffix in _BANNED_EXECUTABLE_SUFFIXES
        or leading.startswith(b"#!")
        or leading.startswith(b"\x7fELF")
        or leading.startswith(b"MZ")
        or leading.startswith((b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"))
    ):
        raise ValueError(f"{label} contains an executable or pickle-capable artifact")


def _require_absolute_real_file(path: Path | str, *, label: str) -> tuple[Path, _FileSnapshot]:
    source = Path(path)
    if not source.is_absolute():
        raise ValueError(f"{label} must be an absolute local path")
    if source.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    snapshot = _stable_file_snapshot(source, label=label)
    _reject_executable_file(source, snapshot, label=label)
    return source.resolve(strict=True), snapshot


def _model_tree_snapshot(path: Path | str) -> tuple[Path, dict[str, Any]]:
    root = Path(path)
    if not root.is_absolute():
        raise ValueError("local model path must be absolute")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("local model path must be one real directory")
    resolved = root.resolve(strict=True)

    def scan() -> dict[str, Any]:
        directories: list[str] = []
        files: list[dict[str, Any]] = []
        for current, raw_directories, raw_files in os.walk(resolved, followlinks=False):
            current_path = Path(current)
            relative_current = current_path.relative_to(resolved)
            for name in sorted(raw_directories):
                child = current_path / name
                state = os.lstat(child)
                if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
                    raise ValueError("local model tree contains a linked or special directory")
                directories.append((relative_current / name).as_posix())
            for name in sorted(raw_files):
                child = current_path / name
                relative = (relative_current / name).as_posix()
                snapshot = _stable_file_snapshot(
                    child,
                    label=f"local model artifact {relative}",
                )
                _reject_executable_file(
                    child,
                    snapshot,
                    label=f"local model artifact {relative}",
                )
                files.append(
                    {
                        "path": relative,
                        "sha256": snapshot.sha256,
                        "size_bytes": snapshot.size_bytes,
                    }
                )
        if not files:
            raise ValueError("local model tree cannot be empty")
        directories.sort()
        files.sort(key=lambda value: value["path"])
        body = {"directories": directories, "files": files}
        return {
            "tree_sha256": _sha256_json(body),
            "file_count": len(files),
            "directory_count": len(directories),
            "total_file_bytes": sum(int(value["size_bytes"]) for value in files),
        }

    first = scan()
    second = scan()
    if first != second:
        raise RuntimeError("local model tree changed while it was being authenticated")
    return resolved, first


def _validated_chunk_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("chunk_configuration must be a mapping")
    raw = copy.deepcopy(dict(value))
    if set(raw) != set(_CHUNK_CONFIG_FIELDS):
        raise ValueError(
            "chunk_configuration must be closed and explicitly include chunk_selection"
        )
    for key in ("chunk_size_words", "chunk_overlap_words", "max_chunks"):
        if isinstance(raw[key], bool) or not isinstance(raw[key], int):
            raise TypeError(f"chunk_configuration.{key} must be an integer")
    if raw["chunk_size_words"] < 1:
        raise ValueError("chunk_size_words must be positive")
    if not 0 <= raw["chunk_overlap_words"] < raw["chunk_size_words"]:
        raise ValueError("chunk_overlap_words must be nonnegative and smaller than chunk size")
    if raw["max_chunks"] < 1:
        raise ValueError("max_chunks must be positive")
    if raw["chunk_selection"] != "last":
        raise ValueError("production embedding cache requires explicit chunk_selection='last'")
    if not isinstance(raw["normalize_embeddings"], bool):
        raise TypeError("normalize_embeddings must be a boolean")
    maximum = raw["max_seq_length"]
    if maximum is not None and (
        isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1
    ):
        raise ValueError("max_seq_length must be null or a positive integer")
    _canonical_json(raw)
    return raw


def _require_nonbinding_chunk_cap(
    texts: Sequence[str],
    *,
    configuration: Mapping[str, Any],
) -> tuple[int, ...]:
    """Prove that ``max_chunks`` cannot discard any source word chunk."""

    chunk_size = int(configuration["chunk_size_words"])
    overlap = int(configuration["chunk_overlap_words"])
    maximum = int(configuration["max_chunks"])
    stride = chunk_size - overlap
    counts: list[int] = []
    offending: list[tuple[int, int]] = []
    for row_id, text in enumerate(texts):
        word_count = sum(1 for _match in re.finditer(r"\S+", str(text or "")))
        uncapped = max(1, int(math.ceil(word_count / stride)))
        counts.append(uncapped)
        if uncapped > maximum:
            offending.append((row_id, uncapped))
    if offending:
        preview = ", ".join(f"row {row_id}: {count} chunks" for row_id, count in offending[:8])
        suffix = "" if len(offending) <= 8 else f", plus {len(offending) - 8} more rows"
        raise ValueError(
            "production embedding max_chunks would cause semantic truncation; "
            f"configured max_chunks={maximum}, uncapped maximum={max(counts)}, "
            f"offending rows: {preview}{suffix}. Raise max_chunks so the cap is nonbinding."
        )
    return tuple(counts)


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key in production embedding cache: {key}")
        output[key] = value
    return output


def _read_json_file(path: Path, *, label: str) -> dict[str, Any]:
    snapshot = _stable_file_snapshot(path, label=label)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    after = _stable_file_snapshot(path, label=label)
    if after.sha256 != snapshot.sha256 or after.signature != snapshot.signature:
        raise RuntimeError(f"{label} changed while it was being decoded")
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _load_dataset_texts(dataset_path: Path, *, text_column: str) -> tuple[str, ...]:
    try:
        frame = pd.read_parquet(dataset_path, columns=[text_column])
    except Exception as exc:
        raise ValueError("cohort Parquet text projection could not be read") from exc
    if list(frame.columns) != [text_column] or len(frame) < 1:
        raise ValueError("cohort text projection is empty or changed its one-column schema")
    values = tuple(frame[text_column].tolist())
    if not all(isinstance(value, str) for value in values):
        raise ValueError("every cohort text row must be an exact non-null string")
    return values


def _ordered_text_sha256(*, text_column: str, texts: Sequence[str]) -> str:
    return _sha256_json(
        {
            "schema_version": "ordered_cohort_text_projection_v1",
            "text_column": text_column,
            "row_count": len(texts),
            "texts": list(texts),
        }
    )


@contextmanager
def _enforced_offline_build() -> Iterator[None]:
    with _OFFLINE_LOCK:
        previous_environment = {key: os.environ.get(key) for key in _OFFLINE_ENVIRONMENT}
        original_create_connection = socket.create_connection
        original_getaddrinfo = socket.getaddrinfo
        original_socket_connect = socket.socket.connect

        def reject_network(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("network access is forbidden while building embedding cache")

        try:
            os.environ.update(_OFFLINE_ENVIRONMENT)
            socket.create_connection = reject_network
            socket.getaddrinfo = reject_network
            socket.socket.connect = reject_network
            yield
        finally:
            socket.create_connection = original_create_connection
            socket.getaddrinfo = original_getaddrinfo
            socket.socket.connect = original_socket_connect
            for key, previous in previous_environment.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous


def _load_local_sentence_encoder(
    *,
    model_path: Path,
    device: str | None,
    max_seq_length: int | None,
) -> Any:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - production dependency
        raise ImportError("sentence-transformers and torch are required") from exc
    encoder = SentenceTransformer(
        str(model_path),
        device=device,
        trust_remote_code=False,
        local_files_only=True,
        model_kwargs={"torch_dtype": torch.float32},
    )
    if max_seq_length is not None:
        current = getattr(encoder, "max_seq_length", None)
        try:
            effective = min(int(current), max_seq_length) if current is not None else max_seq_length
        except (TypeError, ValueError):
            effective = max_seq_length
        encoder.max_seq_length = int(effective)
    encoder.float()
    encoder.eval()
    return encoder


def _effective_max_seq_length(encoder: Any, requested: int | None) -> int | None:
    current = getattr(encoder, "max_seq_length", None)
    try:
        observed = int(current) if current is not None else None
    except (TypeError, ValueError):
        observed = None
    if requested is None:
        return observed if observed is not None and observed > 0 else None
    return min(observed, requested) if observed is not None and observed > 0 else requested


def _default_encoder_prompt(encoder: Any) -> str:
    """Return the exact default prompt that ``SentenceTransformer.encode`` applies."""

    prompt_name = getattr(encoder, "default_prompt_name", None)
    if prompt_name is None:
        return ""
    if not isinstance(prompt_name, str) or not prompt_name:
        raise ValueError("local sentence encoder has an invalid default prompt name")
    prompts = getattr(encoder, "prompts", None)
    if not isinstance(prompts, Mapping) or prompt_name not in prompts:
        raise ValueError("local sentence encoder default prompt is not present in its prompt map")
    prompt = prompts[prompt_name]
    if not isinstance(prompt, str):
        raise ValueError("local sentence encoder default prompt must be an exact string")
    return prompt


def _tokenizer_lengths_without_truncation(tokenizer: Any, inputs: Sequence[str]) -> tuple[int, ...]:
    try:
        encoded = tokenizer(
            list(inputs),
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
    except Exception as exc:
        raise ValueError(
            "local sentence encoder tokenizer could not perform the no-truncation audit"
        ) from exc
    if not isinstance(encoded, Mapping):
        raise ValueError("local sentence encoder tokenizer returned an invalid audit payload")
    raw_lengths = encoded.get("length")
    if raw_lengths is not None:
        if hasattr(raw_lengths, "tolist"):
            raw_lengths = raw_lengths.tolist()
        if isinstance(raw_lengths, int) and not isinstance(raw_lengths, bool):
            raw_lengths = [raw_lengths]
        try:
            lengths = tuple(raw_lengths)
        except TypeError as exc:
            raise ValueError(
                "local sentence encoder tokenizer returned invalid token lengths"
            ) from exc
    else:
        raw_input_ids = encoded.get("input_ids")
        if hasattr(raw_input_ids, "tolist"):
            raw_input_ids = raw_input_ids.tolist()
        if not isinstance(raw_input_ids, Sequence) or isinstance(raw_input_ids, (str, bytes)):
            raise ValueError("local sentence encoder tokenizer omitted auditable token lengths")
        lengths = tuple(
            (
                len(value)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
                else -1
            )
            for value in raw_input_ids
        )
    if len(lengths) != len(inputs) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in lengths
    ):
        raise ValueError("local sentence encoder tokenizer returned invalid token lengths")
    return tuple(int(value) for value in lengths)


def _require_nontruncating_token_lengths(
    *,
    encoder: Any,
    flat_chunks: Sequence[str],
    row_chunk_coordinates: Sequence[tuple[int, int]],
    effective_max_seq_length: int | None,
) -> tuple[int, ...]:
    """Prove every exact encoder input fits without tokenizer-level truncation."""

    if (
        not isinstance(effective_max_seq_length, int)
        or isinstance(effective_max_seq_length, bool)
        or effective_max_seq_length < 1
    ):
        raise ValueError(
            "local sentence encoder must expose a positive effective max sequence length "
            "for the no-truncation audit"
        )
    if len(flat_chunks) < 1 or len(row_chunk_coordinates) != len(flat_chunks):
        raise ValueError("token-length audit requires one row/chunk coordinate per source chunk")
    tokenizer = getattr(encoder, "tokenizer", None)
    if not callable(tokenizer):
        raise ValueError("local sentence encoder does not expose an auditable tokenizer")
    prompt = _default_encoder_prompt(encoder)
    token_counts: list[int] = []
    for start in range(0, len(flat_chunks), _TOKEN_LENGTH_AUDIT_BATCH_SIZE):
        stop = min(start + _TOKEN_LENGTH_AUDIT_BATCH_SIZE, len(flat_chunks))
        token_counts.extend(
            _tokenizer_lengths_without_truncation(
                tokenizer,
                tuple(prompt + chunk for chunk in flat_chunks[start:stop]),
            )
        )
    offending = [
        (flat_index, row_chunk_coordinates[flat_index], count)
        for flat_index, count in enumerate(token_counts)
        if count > effective_max_seq_length
    ]
    if offending:
        preview = ", ".join(
            f"flat chunk {flat_index} (row {coordinate[0]}, row chunk {coordinate[1]}): "
            f"{count} tokens"
            for flat_index, coordinate, count in offending[:8]
        )
        suffix = "" if len(offending) <= 8 else f", plus {len(offending) - 8} more chunks"
        raise ValueError(
            "production embedding tokenizer would cause semantic truncation; "
            f"effective_max_seq_length={effective_max_seq_length}, offending chunks: "
            f"{preview}{suffix}. Re-chunk or repair the source cohort before encoding."
        )
    return tuple(token_counts)


def _write_chunk_registry(path: Path, sample_chunks: Sequence[Sequence[str]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for chunks in sample_chunks:
            handle.write(
                json.dumps(
                    {"chunks": list(chunks)},
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _encode_chunks(
    *,
    encoder: Any,
    flat_chunks: Sequence[str],
    output_path: Path,
    batch_size: int,
    normalize_embeddings: bool,
) -> tuple[int, int]:
    total = len(flat_chunks)
    if total < 1:
        raise ValueError("cohort produced no embedding chunks")
    matrix: np.memmap | None = None
    hidden_size: int | None = None
    cursor = 0
    try:
        while cursor < total:
            stop = min(cursor + batch_size, total)
            encoded = encoder.encode(
                list(flat_chunks[cursor:stop]),
                batch_size=stop - cursor,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
            )
            values = np.asarray(encoded, dtype=np.float32)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            if (
                values.ndim != 2
                or values.shape[0] != stop - cursor
                or values.shape[1] < 1
                or not np.isfinite(values).all()
            ):
                raise ValueError("local sentence encoder returned an invalid embedding matrix")
            if hidden_size is None:
                hidden_size = int(values.shape[1])
                matrix = np.lib.format.open_memmap(
                    output_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(total, hidden_size),
                )
            elif int(values.shape[1]) != hidden_size:
                raise ValueError("local sentence encoder changed embedding dimension")
            assert matrix is not None
            matrix[cursor:stop] = values
            cursor = stop
        assert matrix is not None and hidden_size is not None
        matrix.flush()
    finally:
        if matrix is not None:
            del matrix
    return total, int(hidden_size or 0)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cache_file_registrations(root: Path) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("embedding cache must be one real directory")
    candidates = list(root.iterdir())
    if any(path.is_symlink() for path in candidates) or {path.name for path in candidates} != set(
        _CACHE_FILES
    ):
        raise ValueError("embedding cache must contain exactly its four non-linked data files")
    registrations: dict[str, dict[str, Any]] = {}
    for name in sorted(_CACHE_FILES):
        path = root / name
        snapshot = _stable_file_snapshot(path, label=f"embedding cache {name}")
        _reject_executable_file(path, snapshot, label=f"embedding cache {name}")
        registrations[name] = {
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
        }
    return registrations


def _validate_closed_provenance_shapes(provenance: Mapping[str, Any]) -> None:
    dataset = provenance["dataset"]
    model = provenance["local_model"]
    execution = provenance["encoder_execution"]
    companions = provenance["companion_cache_files"]
    dataset_path = dataset.get("path")
    model_path = model.get("path")
    if (
        not isinstance(dataset_path, str)
        or not Path(dataset_path).is_absolute()
        or not isinstance(dataset.get("size_bytes"), int)
        or isinstance(dataset.get("size_bytes"), bool)
        or dataset["size_bytes"] < 0
        or not isinstance(dataset.get("text_column"), str)
        or not dataset["text_column"].strip()
        or not isinstance(dataset.get("row_count"), int)
        or isinstance(dataset.get("row_count"), bool)
        or dataset["row_count"] < 1
        or not isinstance(model_path, str)
        or not Path(model_path).is_absolute()
        or not isinstance(model.get("file_count"), int)
        or isinstance(model.get("file_count"), bool)
        or model["file_count"] < 1
        or not isinstance(model.get("directory_count"), int)
        or isinstance(model.get("directory_count"), bool)
        or model["directory_count"] < 0
        or not isinstance(model.get("total_file_bytes"), int)
        or isinstance(model.get("total_file_bytes"), bool)
        or model["total_file_bytes"] < 0
    ):
        raise ValueError("production embedding cache provenance has invalid dataset/model fields")
    for field_name, source in (
        ("dataset.sha256", dataset.get("sha256")),
        ("dataset.ordered_text_sha256", dataset.get("ordered_text_sha256")),
        ("local_model.tree_sha256", model.get("tree_sha256")),
    ):
        _require_sha256(source, field_name=field_name)
    device = execution.get("device")
    if (
        (
            device is not None
            and (
                not isinstance(device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None
            )
        )
        or not isinstance(execution.get("batch_size"), int)
        or isinstance(execution.get("batch_size"), bool)
        or execution["batch_size"] < 1
        or execution.get("local_files_only") is not True
        or execution.get("trust_remote_code") is not False
        or execution.get("offline_environment") != _OFFLINE_ENVIRONMENT
        or execution.get("socket_access_blocked") is not True
    ):
        raise ValueError("production embedding cache provenance has invalid encoder policy")
    for name, registration in companions.items():
        size = registration.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"production embedding cache {name} has an invalid size")
        _require_sha256(
            registration.get("sha256"),
            field_name=f"companion_cache_files.{name}.sha256",
        )


def _validate_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_sentence_model_name: str,
    expected_configuration: Mapping[str, Any],
    expected_rows: int,
    expected_companions: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(metadata) != set(_METADATA_FIELDS):
        raise ValueError("production embedding cache metadata is not a closed schema")
    provenance = metadata.get("production_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(_PROVENANCE_FIELDS):
        raise ValueError("production embedding cache provenance is not a closed schema")
    dataset = provenance.get("dataset")
    model = provenance.get("local_model")
    execution = provenance.get("encoder_execution")
    companions = provenance.get("companion_cache_files")
    if (
        not isinstance(dataset, Mapping)
        or set(dataset) != set(_DATASET_PROVENANCE_FIELDS)
        or not isinstance(model, Mapping)
        or set(model) != set(_MODEL_PROVENANCE_FIELDS)
        or not isinstance(execution, Mapping)
        or set(execution) != set(_ENCODER_EXECUTION_FIELDS)
        or not isinstance(companions, Mapping)
        or set(companions) != set(_COMPANION_FILES)
        or any(
            not isinstance(value, Mapping) or set(value) != set(_FILE_REGISTRATION_FIELDS)
            for value in companions.values()
        )
    ):
        raise ValueError("production embedding cache provenance has an open nested schema")
    _validate_closed_provenance_shapes(provenance)

    hidden_size = metadata.get("hidden_size")
    num_samples = metadata.get("num_samples")
    total_chunks = metadata.get("total_chunks")
    chunk_counts = metadata.get("chunk_counts")
    if (
        not isinstance(hidden_size, int)
        or isinstance(hidden_size, bool)
        or hidden_size < 1
        or not isinstance(num_samples, int)
        or isinstance(num_samples, bool)
        or num_samples != expected_rows
        or not isinstance(total_chunks, int)
        or isinstance(total_chunks, bool)
        or total_chunks < 1
        or not isinstance(chunk_counts, list)
        or len(chunk_counts) != expected_rows
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in chunk_counts
        )
        or sum(chunk_counts) != total_chunks
    ):
        raise ValueError("production embedding cache metadata has invalid dimensions")

    metadata_configuration = {key: metadata.get(key) for key in _CHUNK_CONFIG_FIELDS}
    try:
        validated_metadata_configuration = _validated_chunk_configuration(metadata_configuration)
        validated_provenance_configuration = _validated_chunk_configuration(
            provenance.get("chunk_configuration")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "production embedding cache metadata changed its authenticated policy"
        ) from exc
    requested_max_seq_length = expected_configuration["max_seq_length"]
    effective_max_seq_length = metadata.get("effective_max_seq_length")
    if requested_max_seq_length is None:
        valid_effective_max_seq_length = (
            isinstance(effective_max_seq_length, int)
            and not isinstance(effective_max_seq_length, bool)
            and effective_max_seq_length >= 1
        )
    else:
        valid_effective_max_seq_length = (
            isinstance(effective_max_seq_length, int)
            and not isinstance(effective_max_seq_length, bool)
            and 1 <= effective_max_seq_length <= requested_max_seq_length
        )
    max_observed_token_count = metadata.get("max_observed_token_count")
    valid_token_audit = (
        isinstance(max_observed_token_count, int)
        and not isinstance(max_observed_token_count, bool)
        and max_observed_token_count >= 1
        and isinstance(effective_max_seq_length, int)
        and max_observed_token_count <= effective_max_seq_length
    )

    if (
        metadata.get("schema_version") != PRODUCTION_EMBEDDING_CACHE_METADATA_SCHEMA
        or metadata.get("sentence_model_name") != expected_sentence_model_name
        or validated_metadata_configuration != dict(expected_configuration)
        or _canonical_json(validated_metadata_configuration)
        != _canonical_json(dict(expected_configuration))
        or not valid_effective_max_seq_length
        or metadata.get("chunking_mode")
        != "whitespace_word_chunks_tokenizer_verified_nontruncating_v2"
        or metadata.get("actual_max_len") != max(chunk_counts)
        or not isinstance(metadata.get("actual_max_len"), int)
        or isinstance(metadata.get("actual_max_len"), bool)
        or max(chunk_counts) > int(expected_configuration["max_chunks"])
        or metadata.get("uncapped_total_chunks") != total_chunks
        or metadata.get("uncapped_chunk_counts_sha256") != _sha256_json(chunk_counts)
        or metadata.get("chunk_cap_nonbinding") is not True
        or metadata.get("semantic_truncation_allowed") is not False
        or not valid_token_audit
        or metadata.get("tokenizer_truncation_allowed") is not False
        or metadata.get("storage_format") != "variable_length_chunks"
        or metadata.get("dtype") != "float32"
        or provenance.get("schema_version") != PRODUCTION_EMBEDDING_CACHE_PROVENANCE_SCHEMA
        or provenance.get("builder_version") != PRODUCTION_EMBEDDING_CACHE_BUILDER_VERSION
        or provenance.get("builder_code_sha256") != _builder_code_sha256()
        or provenance.get("sentence_model_name") != expected_sentence_model_name
        or validated_provenance_configuration != dict(expected_configuration)
        or _canonical_json(validated_provenance_configuration)
        != _canonical_json(dict(expected_configuration))
        or provenance.get("chunk_configuration_sha256") != _sha256_json(expected_configuration)
        or provenance.get("cache_configuration_sha256")
        != _cache_configuration_sha256(
            sentence_model_name=expected_sentence_model_name,
            chunk_configuration=expected_configuration,
        )
        or companions != dict(expected_companions)
        or provenance.get("uncapped_total_chunks") != total_chunks
        or provenance.get("uncapped_chunk_counts_sha256") != _sha256_json(chunk_counts)
        or provenance.get("chunk_cap_nonbinding") is not True
        or provenance.get("semantic_truncation_allowed") is not False
        or provenance.get("max_observed_token_count") != max_observed_token_count
        or provenance.get("ordered_token_counts_sha256")
        != metadata.get("ordered_token_counts_sha256")
        or provenance.get("tokenizer_truncation_allowed") is not False
        or provenance.get("atomic_publication") != "fresh_temp_sibling_directory_rename_v1"
        or provenance.get("partial_cache_reuse_allowed") is not False
        or provenance.get("network_access_allowed") is not False
        or provenance.get("symlinks_allowed") is not False
        or provenance.get("executable_artifacts_allowed") is not False
        or metadata.get("production_provenance_sha256") != _sha256_json(provenance)
    ):
        raise ValueError("production embedding cache metadata changed its authenticated policy")
    for field_name in (
        "production_provenance_sha256",
        "builder_code_sha256",
        "chunk_configuration_sha256",
        "cache_configuration_sha256",
    ):
        source = metadata if field_name == "production_provenance_sha256" else provenance
        _require_sha256(source.get(field_name), field_name=field_name)
    metadata_token_counts_sha256 = _require_sha256(
        metadata.get("ordered_token_counts_sha256"),
        field_name="ordered_token_counts_sha256",
    )
    provenance_token_counts_sha256 = _require_sha256(
        provenance.get("ordered_token_counts_sha256"),
        field_name="provenance.ordered_token_counts_sha256",
    )
    if metadata_token_counts_sha256 != provenance_token_counts_sha256:
        raise ValueError("production embedding cache token-length proofs differ")


def _validate_cache_files_against_provider_identity(
    cache_files: Any,
    provider_identity: Any,
    *,
    expected_rows: int,
    expected_chunks: int,
) -> None:
    if not isinstance(cache_files, Mapping) or set(cache_files) != set(_CACHE_FILES):
        raise ValueError("production embedding cache file identity is not closed")
    if not isinstance(provider_identity, Mapping) or set(provider_identity) != set(
        _PROVIDER_IDENTITY_FIELDS
    ):
        raise ValueError("production embedding cache provider identity is not closed")
    if (
        provider_identity.get("provider") != "spent_only_frozen_chunk_embedding_cache_v2"
        or not isinstance(provider_identity.get("row_count"), int)
        or isinstance(provider_identity.get("row_count"), bool)
        or provider_identity.get("row_count") != expected_rows
        or not isinstance(provider_identity.get("chunk_count"), int)
        or isinstance(provider_identity.get("chunk_count"), bool)
        or provider_identity.get("chunk_count") != expected_chunks
        or provider_identity.get("cache_snapshot_authentication") != "streamed_private_fd_sha256_v1"
        or provider_identity.get("chunk_text_storage") != "private_fd_pread_lazy_row_decode_v1"
        or provider_identity.get("embeddings_path_backed") is not False
        or provider_identity.get("private_snapshot_embedding_mmap") is not True
        or provider_identity.get("future_row_text_decoded") is not False
        or provider_identity.get("novel_text_encoding_allowed") is not False
    ):
        raise ValueError("production embedding cache provider identity changed its policy")
    for name, hash_field in _PROVIDER_CACHE_HASH_FIELDS.items():
        registration = cache_files.get(name)
        if (
            not isinstance(registration, Mapping)
            or set(registration) != set(_FILE_REGISTRATION_FIELDS)
            or not isinstance(registration.get("size_bytes"), int)
            or isinstance(registration.get("size_bytes"), bool)
            or registration["size_bytes"] < 1
        ):
            raise ValueError(f"production embedding cache registration is invalid: {name}")
        registered_hash = _require_sha256(
            registration.get("sha256"),
            field_name=f"cache_files.{name}.sha256",
        )
        provider_hash = _require_sha256(
            provider_identity.get(hash_field),
            field_name=f"provider_identity.{hash_field}",
        )
        if registered_hash != provider_hash:
            raise ValueError(
                f"production embedding cache registration differs from provider: {name}"
            )


def _validate_cache_content(
    root: Path,
    *,
    texts: tuple[str, ...],
    sentence_model_name: str,
    configuration: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], Mapping[str, Any], dict[str, Any]]:
    uncapped_chunk_counts = _require_nonbinding_chunk_cap(
        texts,
        configuration=configuration,
    )
    root_signature = _real_directory_signature(root, label="embedding cache root")
    registrations = _cache_file_registrations(root)
    metadata = _read_json_file(root / "metadata.json", label="embedding cache metadata")
    companions = {name: registrations[name] for name in _COMPANION_FILES}
    _validate_metadata(
        metadata,
        expected_sentence_model_name=sentence_model_name,
        expected_configuration=configuration,
        expected_rows=len(texts),
        expected_companions=companions,
    )
    if tuple(metadata["chunk_counts"]) != uncapped_chunk_counts:
        raise ValueError(
            "production embedding cache chunk registry is not the exact uncapped source projection"
        )
    try:
        embeddings = np.load(root / "chunk_embeddings.npy", mmap_mode="r", allow_pickle=False)
        offsets = np.load(root / "offsets.npy", allow_pickle=False)
    except Exception as exc:
        raise ValueError("embedding cache arrays are invalid or pickle-capable") from exc
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] < 1
        or embeddings.shape[1] < 1
        or embeddings.dtype != np.dtype(np.float32)
        or not np.isfinite(embeddings).all()
        or offsets.dtype != np.int64
        or offsets.ndim != 1
        or len(offsets) != len(texts) + 1
        or int(offsets[0]) != 0
        or np.any(np.diff(offsets) < 1)
        or int(offsets[-1]) != int(embeddings.shape[0])
        or int(metadata["hidden_size"]) != int(embeddings.shape[1])
        or int(metadata["total_chunks"]) != int(embeddings.shape[0])
        or list(metadata["chunk_counts"]) != np.diff(offsets).astype(int).tolist()
    ):
        raise ValueError("embedding cache arrays do not match their closed metadata")
    cache = SpentOnlyFrozenChunkEmbeddingCache(root)
    provider = cache.bind_spent(tuple(range(len(texts))), texts)
    provider_identity = provider.identity()
    if cache.row_count != len(texts) or tuple(provider.row_ids) != tuple(range(len(texts))):
        raise RuntimeError("production embedding cache failed all-row binding")
    final_registrations = _cache_file_registrations(root)
    final_metadata = _read_json_file(root / "metadata.json", label="embedding cache metadata")
    if final_registrations != registrations or final_metadata != metadata:
        raise RuntimeError("production embedding cache changed while it was being validated")
    if _real_directory_signature(root, label="embedding cache root") != root_signature:
        raise RuntimeError("production embedding cache root changed while it was being validated")
    cache_identity = provider_identity["cache"]
    _validate_cache_files_against_provider_identity(
        final_registrations,
        cache_identity,
        expected_rows=len(texts),
        expected_chunks=int(embeddings.shape[0]),
    )
    return final_registrations, cache_identity, copy.deepcopy(metadata)


def _same_file_snapshot(first: _FileSnapshot, second: _FileSnapshot) -> bool:
    return (
        first.sha256 == second.sha256
        and first.size_bytes == second.size_bytes
        and first.signature == second.signature
    )


def _owned_directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    state = os.lstat(path)
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise ValueError(f"{label} must be one real directory")
    return int(state.st_dev), int(state.st_ino)


def _safe_remove_owned_directory(
    path: Path,
    *,
    parent: Path,
    prefix: str,
    expected_identity: tuple[int, int],
) -> bool:
    """Remove only the exact directory object created by this build.

    The path can move once, from its temporary sibling name to the publication
    target.  Device/inode binding prevents cleanup from deleting a path that an
    external process populated or substituted after the build started.
    """

    target = Path(path)
    if target.parent != parent or not target.name.startswith(prefix):
        raise RuntimeError("refusing to clean an unowned embedding-cache build path")
    try:
        state = os.lstat(target)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise RuntimeError("refusing to clean a replaced embedding-cache build path")
    observed_identity = (int(state.st_dev), int(state.st_ino))
    if observed_identity != expected_identity:
        raise RuntimeError("refusing to clean an embedding-cache path not owned by this build")
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise RuntimeError("safe embedding-cache cleanup is unavailable on this platform")
    shutil.rmtree(target)
    if os.path.lexists(target):
        raise RuntimeError("owned embedding-cache build path remains after cleanup")
    return True


@dataclass(frozen=True)
class ProductionEmbeddingCacheBuildResult:
    """Published cache path plus a detached, closed integration identity."""

    cache_path: Path
    _identity: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        path = Path(self.cache_path)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("published embedding cache path must be one real directory")
        resolved = path.resolve(strict=True)
        identity = copy.deepcopy(dict(self._identity))
        if set(identity) != set(_RESULT_FIELDS):
            raise ValueError("production embedding cache result identity is not closed")
        if (
            identity.get("schema_version") != PRODUCTION_EMBEDDING_CACHE_RESULT_SCHEMA
            or identity.get("builder_version") != PRODUCTION_EMBEDDING_CACHE_BUILDER_VERSION
            or identity.get("builder_code_sha256") != _builder_code_sha256()
            or _validated_sentence_model_name(identity.get("sentence_model_name"))
            != identity.get("sentence_model_name")
            or identity.get("cache_path") != str(resolved)
            or identity.get("atomic_publication") != "fresh_temp_sibling_directory_rename_v1"
            or identity.get("offline_build") is not True
        ):
            raise ValueError("production embedding cache result changed its security policy")
        for field_name in (
            "builder_code_sha256",
            "production_provenance_sha256",
            "dataset_sha256",
            "ordered_text_sha256",
            "local_model_tree_sha256",
            "chunk_configuration_sha256",
            "cache_configuration_sha256",
        ):
            _require_sha256(identity.get(field_name), field_name=f"result.{field_name}")
        for field_name in ("row_count", "chunk_count", "hidden_size"):
            value = identity.get(field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"production embedding cache result has invalid {field_name}")
        actual_files = _cache_file_registrations(resolved)
        if identity.get("cache_files") != actual_files:
            raise ValueError("production embedding cache result differs from published file bytes")
        _validate_cache_files_against_provider_identity(
            actual_files,
            identity.get("provider_identity"),
            expected_rows=identity["row_count"],
            expected_chunks=identity["chunk_count"],
        )
        object.__setattr__(self, "cache_path", resolved)
        object.__setattr__(self, "_identity", MappingProxyType(identity))

    def identity(self) -> Mapping[str, Any]:
        if _builder_code_sha256() != self._identity["builder_code_sha256"]:
            raise RuntimeError("production embedding cache builder code changed")
        if _cache_file_registrations(self.cache_path) != self._identity["cache_files"]:
            raise RuntimeError("production embedding cache bytes changed")
        return copy.deepcopy(dict(self._identity))


def _result_identity_from_validated_cache(
    *,
    cache_path: Path,
    metadata: Mapping[str, Any],
    cache_files: Mapping[str, Mapping[str, Any]],
    provider_identity: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = metadata["production_provenance"]
    dataset = provenance["dataset"]
    model = provenance["local_model"]
    return {
        "schema_version": PRODUCTION_EMBEDDING_CACHE_RESULT_SCHEMA,
        "builder_version": PRODUCTION_EMBEDDING_CACHE_BUILDER_VERSION,
        "builder_code_sha256": provenance["builder_code_sha256"],
        "cache_path": str(cache_path.resolve(strict=True)),
        "production_provenance_sha256": metadata["production_provenance_sha256"],
        "dataset_sha256": dataset["sha256"],
        "ordered_text_sha256": dataset["ordered_text_sha256"],
        "sentence_model_name": provenance["sentence_model_name"],
        "local_model_tree_sha256": model["tree_sha256"],
        "chunk_configuration_sha256": provenance["chunk_configuration_sha256"],
        "cache_configuration_sha256": provenance["cache_configuration_sha256"],
        "row_count": int(metadata["num_samples"]),
        "chunk_count": int(metadata["total_chunks"]),
        "hidden_size": int(metadata["hidden_size"]),
        "cache_files": copy.deepcopy(dict(cache_files)),
        "provider_identity": copy.deepcopy(dict(provider_identity)),
        "atomic_publication": "fresh_temp_sibling_directory_rename_v1",
        "offline_build": True,
    }


def _build_production_embedding_cache(
    *,
    dataset_path: Path | str,
    text_column: str,
    local_model_path: Path | str,
    sentence_model_name: str,
    chunk_configuration: Mapping[str, Any],
    target_dir: Path | str,
    device: str | None = None,
    batch_size: int = 32,
) -> ProductionEmbeddingCacheBuildResult:
    """Build and atomically publish one authenticated arbitrary-cohort cache."""

    if not isinstance(text_column, str) or not text_column.strip():
        raise ValueError("text_column must be a non-empty exact column name")
    logical_model_name = _validated_sentence_model_name(sentence_model_name)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if device is not None and (
        not isinstance(device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None
    ):
        raise ValueError("device must be null, cpu, cuda, or one explicit cuda index")
    supplied_chunk_configuration = chunk_configuration
    configuration = _validated_chunk_configuration(supplied_chunk_configuration)

    target = Path(target_dir)
    if not target.is_absolute():
        raise ValueError("target_dir must be an absolute path")
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("target_dir parent must be one existing real directory")
    if target.is_symlink() or target.exists():
        raise FileExistsError("production embedding cache target must be fresh and non-symlink")

    dataset, dataset_before = _require_absolute_real_file(
        dataset_path,
        label="cohort dataset",
    )
    texts = _load_dataset_texts(dataset, text_column=text_column)
    dataset_after_read = _stable_file_snapshot(dataset, label="cohort dataset")
    if not _same_file_snapshot(dataset_before, dataset_after_read):
        raise RuntimeError("cohort dataset changed while its text projection was read")
    ordered_text_sha256 = _ordered_text_sha256(text_column=text_column, texts=texts)
    uncapped_chunk_counts = _require_nonbinding_chunk_cap(
        texts,
        configuration=configuration,
    )
    model, model_before = _model_tree_snapshot(local_model_path)

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.building-",
            dir=parent,
        )
    )
    temporary_prefix = f".{target.name}.building-"
    owned_directory_identity = _owned_directory_identity(
        temporary,
        label="temporary embedding-cache build root",
    )
    try:
        sample_chunks = tuple(
            tuple(
                chunk_text_words(
                    text,
                    configuration["chunk_size_words"],
                    configuration["chunk_overlap_words"],
                    configuration["max_chunks"],
                    configuration["chunk_selection"],
                )
            )
            for text in texts
        )
        chunk_counts = [len(chunks) for chunks in sample_chunks]
        if tuple(chunk_counts) != uncapped_chunk_counts:
            raise RuntimeError("production word chunker differs from its non-truncation preflight")
        offsets = np.zeros(len(texts) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(np.asarray(chunk_counts, dtype=np.int64))
        np.save(temporary / "offsets.npy", offsets, allow_pickle=False)
        _write_chunk_registry(temporary / "chunk_texts.jsonl", sample_chunks)
        flat_chunks = tuple(chunk for chunks in sample_chunks for chunk in chunks)
        row_chunk_coordinates = tuple(
            (row_index, chunk_index)
            for row_index, chunks in enumerate(sample_chunks)
            for chunk_index, _chunk in enumerate(chunks)
        )

        encoder = _load_local_sentence_encoder(
            model_path=model,
            device=device,
            max_seq_length=configuration["max_seq_length"],
        )
        effective_max_seq_length = _effective_max_seq_length(
            encoder,
            configuration["max_seq_length"],
        )
        ordered_token_counts = _require_nontruncating_token_lengths(
            encoder=encoder,
            flat_chunks=flat_chunks,
            row_chunk_coordinates=row_chunk_coordinates,
            effective_max_seq_length=effective_max_seq_length,
        )
        total_chunks, hidden_size = _encode_chunks(
            encoder=encoder,
            flat_chunks=flat_chunks,
            output_path=temporary / "chunk_embeddings.npy",
            batch_size=batch_size,
            normalize_embeddings=configuration["normalize_embeddings"],
        )

        companion_registrations = {
            name: {
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
            }
            for name in _COMPANION_FILES
            for snapshot in (
                _stable_file_snapshot(
                    temporary / name,
                    label=f"generated embedding cache {name}",
                ),
            )
        }
        dataset_provenance = {
            "path": str(dataset),
            "sha256": dataset_before.sha256,
            "size_bytes": dataset_before.size_bytes,
            "text_column": text_column,
            "row_count": len(texts),
            "ordered_text_sha256": ordered_text_sha256,
        }
        model_provenance = {"path": str(model), **model_before}
        encoder_execution = {
            "device": device,
            "batch_size": batch_size,
            "local_files_only": True,
            "trust_remote_code": False,
            "offline_environment": copy.deepcopy(_OFFLINE_ENVIRONMENT),
            "socket_access_blocked": True,
        }
        provenance = {
            "schema_version": PRODUCTION_EMBEDDING_CACHE_PROVENANCE_SCHEMA,
            "builder_version": PRODUCTION_EMBEDDING_CACHE_BUILDER_VERSION,
            "builder_code_sha256": _builder_code_sha256(),
            "dataset": dataset_provenance,
            "sentence_model_name": logical_model_name,
            "local_model": model_provenance,
            "chunk_configuration": copy.deepcopy(configuration),
            "chunk_configuration_sha256": _sha256_json(configuration),
            "cache_configuration_sha256": _cache_configuration_sha256(
                sentence_model_name=logical_model_name,
                chunk_configuration=configuration,
            ),
            "encoder_execution": encoder_execution,
            "companion_cache_files": companion_registrations,
            "uncapped_total_chunks": sum(uncapped_chunk_counts),
            "uncapped_chunk_counts_sha256": _sha256_json(uncapped_chunk_counts),
            "chunk_cap_nonbinding": True,
            "semantic_truncation_allowed": False,
            "max_observed_token_count": max(ordered_token_counts),
            "ordered_token_counts_sha256": _sha256_json(ordered_token_counts),
            "tokenizer_truncation_allowed": False,
            "atomic_publication": "fresh_temp_sibling_directory_rename_v1",
            "partial_cache_reuse_allowed": False,
            "network_access_allowed": False,
            "symlinks_allowed": False,
            "executable_artifacts_allowed": False,
        }
        metadata = {
            "schema_version": PRODUCTION_EMBEDDING_CACHE_METADATA_SCHEMA,
            "sentence_model_name": logical_model_name,
            "hidden_size": hidden_size,
            "num_samples": len(texts),
            "total_chunks": total_chunks,
            "chunk_counts": chunk_counts,
            **copy.deepcopy(configuration),
            "effective_max_seq_length": effective_max_seq_length,
            "chunking_mode": "whitespace_word_chunks_tokenizer_verified_nontruncating_v2",
            "actual_max_len": max(chunk_counts),
            "uncapped_total_chunks": sum(uncapped_chunk_counts),
            "uncapped_chunk_counts_sha256": _sha256_json(uncapped_chunk_counts),
            "chunk_cap_nonbinding": True,
            "semantic_truncation_allowed": False,
            "max_observed_token_count": max(ordered_token_counts),
            "ordered_token_counts_sha256": _sha256_json(ordered_token_counts),
            "tokenizer_truncation_allowed": False,
            "storage_format": "variable_length_chunks",
            "dtype": "float32",
            "production_provenance": provenance,
            "production_provenance_sha256": _sha256_json(provenance),
        }
        _write_json_new(temporary / "metadata.json", metadata)
        (
            before_publish_files,
            before_provider_identity,
            before_metadata,
        ) = _validate_cache_content(
            temporary,
            texts=texts,
            sentence_model_name=logical_model_name,
            configuration=configuration,
        )

        dataset_after = _stable_file_snapshot(dataset, label="cohort dataset")
        _model_after_path, model_after = _model_tree_snapshot(model)
        configuration_after = _validated_chunk_configuration(supplied_chunk_configuration)
        if not _same_file_snapshot(dataset_before, dataset_after):
            raise RuntimeError("cohort dataset changed during embedding-cache build")
        if model_after != model_before:
            raise RuntimeError("local model tree changed during embedding-cache build")
        if configuration_after != configuration:
            raise RuntimeError("chunk configuration changed during embedding-cache build")
        if target.is_symlink() or target.exists():
            raise FileExistsError("production embedding cache target was populated during build")

        os.rename(temporary, target)
        after_publish_files, provider_identity, after_metadata = _validate_cache_content(
            target,
            texts=texts,
            sentence_model_name=logical_model_name,
            configuration=configuration,
        )
        if (
            after_publish_files != before_publish_files
            or provider_identity != before_provider_identity
            or after_metadata != before_metadata
        ):
            raise RuntimeError("published embedding cache differs from validated temporary cache")
        dataset_final = _stable_file_snapshot(dataset, label="cohort dataset")
        _model_final_path, model_final = _model_tree_snapshot(model)
        if not _same_file_snapshot(dataset_before, dataset_final) or model_final != model_before:
            raise RuntimeError("input or model changed while embedding cache was published")

        result_identity = _result_identity_from_validated_cache(
            cache_path=target,
            metadata=after_metadata,
            cache_files=after_publish_files,
            provider_identity=provider_identity,
        )
        return ProductionEmbeddingCacheBuildResult(
            cache_path=target,
            _identity=result_identity,
        )
    except BaseException as build_error:
        # Signals can arrive immediately after the atomic rename and before the
        # next Python bytecode executes. Check both allowed names, but stop once
        # the exact device/inode owned by this build has been removed. Cleanup
        # failures are attached diagnostically and never replace the triggering
        # KeyboardInterrupt, SystemExit, or ordinary exception.
        removed = False
        for cleanup_path, cleanup_prefix in (
            (temporary, temporary_prefix),
            (target, target.name),
        ):
            if removed:
                break
            try:
                removed = _safe_remove_owned_directory(
                    cleanup_path,
                    parent=parent,
                    prefix=cleanup_prefix,
                    expected_identity=owned_directory_identity,
                )
            except BaseException as cleanup_error:
                try:
                    build_error.add_note(
                        "embedding-cache cleanup could not remove "
                        f"{cleanup_path}: {type(cleanup_error).__name__}: {cleanup_error}"
                    )
                except BaseException:
                    pass
        raise


def build_production_embedding_cache(
    *,
    dataset_path: Path | str,
    text_column: str,
    local_model_path: Path | str,
    sentence_model_name: str,
    chunk_configuration: Mapping[str, Any],
    target_dir: Path | str,
    device: str | None = None,
    batch_size: int = 32,
) -> ProductionEmbeddingCacheBuildResult:
    """Build under process-wide offline guards and atomically publish."""

    with _enforced_offline_build():
        return _build_production_embedding_cache(
            dataset_path=dataset_path,
            text_column=text_column,
            local_model_path=local_model_path,
            sentence_model_name=sentence_model_name,
            chunk_configuration=chunk_configuration,
            target_dir=target_dir,
            device=device,
            batch_size=batch_size,
        )


def validate_published_production_embedding_cache(
    *,
    cache_dir: Path | str,
    dataset_path: Path | str,
    text_column: str,
    sentence_model_name: str,
    chunk_configuration: Mapping[str, Any],
    expected_local_model_path: Path | str | None = None,
) -> Mapping[str, Any]:
    """Read-only authentication of one already published production cache.

    The returned mapping has the same closed schema as
    ``build_production_embedding_cache(...).identity()``. The encoder is never
    constructed; an optional local model path is authenticated as a byte tree.
    """

    if not isinstance(text_column, str) or not text_column.strip():
        raise ValueError("text_column must be a non-empty exact column name")
    logical_model_name = _validated_sentence_model_name(sentence_model_name)
    supplied_chunk_configuration = chunk_configuration
    configuration = _validated_chunk_configuration(supplied_chunk_configuration)

    supplied_cache = Path(cache_dir)
    if not supplied_cache.is_absolute():
        raise ValueError("cache_dir must be an absolute path")
    if supplied_cache.is_symlink() or not supplied_cache.is_dir():
        raise ValueError("cache_dir must be one existing real directory")
    cache = supplied_cache.resolve(strict=True)
    if cache != supplied_cache:
        raise ValueError("cache_dir cannot contain symlinked or non-canonical path components")
    cache_root_before = _real_directory_signature(cache, label="embedding cache root")

    dataset, dataset_before = _require_absolute_real_file(
        dataset_path,
        label="cohort dataset",
    )
    texts = _load_dataset_texts(dataset, text_column=text_column)
    dataset_after_read = _stable_file_snapshot(dataset, label="cohort dataset")
    if not _same_file_snapshot(dataset_before, dataset_after_read):
        raise RuntimeError("cohort dataset changed while its text projection was read")
    ordered_text_sha256 = _ordered_text_sha256(text_column=text_column, texts=texts)
    _require_nonbinding_chunk_cap(texts, configuration=configuration)
    expected_dataset_provenance = {
        "path": str(dataset),
        "sha256": dataset_before.sha256,
        "size_bytes": dataset_before.size_bytes,
        "text_column": text_column,
        "row_count": len(texts),
        "ordered_text_sha256": ordered_text_sha256,
    }

    expected_model: Path | None = None
    model_before: dict[str, Any] | None = None
    if expected_local_model_path is not None:
        expected_model, model_before = _model_tree_snapshot(expected_local_model_path)

    cache_files, provider_identity, metadata = _validate_cache_content(
        cache,
        texts=texts,
        sentence_model_name=logical_model_name,
        configuration=configuration,
    )
    provenance = metadata["production_provenance"]
    if provenance["dataset"] != expected_dataset_provenance:
        raise ValueError("production embedding cache provenance differs from the supplied cohort")
    if expected_model is not None and model_before is not None:
        expected_model_provenance = {"path": str(expected_model), **model_before}
        if provenance["local_model"] != expected_model_provenance:
            raise ValueError(
                "production embedding cache provenance differs from the supplied local model"
            )

    dataset_final = _stable_file_snapshot(dataset, label="cohort dataset")
    if not _same_file_snapshot(dataset_before, dataset_final):
        raise RuntimeError("cohort dataset changed while its cache was being validated")
    configuration_final = _validated_chunk_configuration(supplied_chunk_configuration)
    if configuration_final != configuration:
        raise RuntimeError("chunk configuration changed while cache was being validated")
    if expected_model is not None and model_before is not None:
        model_final_path, model_final = _model_tree_snapshot(expected_model)
        if model_final_path != expected_model or model_final != model_before:
            raise RuntimeError("local model tree changed while cache was being validated")
    if _real_directory_signature(cache, label="embedding cache root") != cache_root_before:
        raise RuntimeError("production embedding cache root changed while it was being validated")

    result = ProductionEmbeddingCacheBuildResult(
        cache_path=cache,
        _identity=_result_identity_from_validated_cache(
            cache_path=cache,
            metadata=metadata,
            cache_files=cache_files,
            provider_identity=provider_identity,
        ),
    )
    identity = result.identity()
    if _real_directory_signature(cache, label="embedding cache root") != cache_root_before:
        raise RuntimeError("production embedding cache root changed while it was being validated")
    return identity


__all__ = [
    "PRODUCTION_EMBEDDING_CACHE_BUILDER_VERSION",
    "PRODUCTION_EMBEDDING_CACHE_METADATA_SCHEMA",
    "PRODUCTION_EMBEDDING_CACHE_PROVENANCE_SCHEMA",
    "PRODUCTION_EMBEDDING_CACHE_RESULT_SCHEMA",
    "ProductionEmbeddingCacheBuildResult",
    "build_production_embedding_cache",
    "validate_published_production_embedding_cache",
]
