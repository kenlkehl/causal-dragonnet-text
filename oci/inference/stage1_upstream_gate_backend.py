"""Gate-honest adapter for the historical all-source Stage-1 numerical bank.

The integrated Stage-1 runner already knows how to fit BoW nuisance/R models,
HTR nuisance/R models, matched-pair uplift models, and whole/cluster embedding
contrasts.  Historical outer-fold matrices cannot be reused for an adaptive
review gate because their inner folds do not recursively exclude that entire
gate.  This adapter instead invokes the existing builder with the current
spent context as its train frame and a label-free gate frame as its test frame.

Frozen chunk embeddings are served through a row-bound cache view.  Constructing
the backend authenticates only cache bytes and row offsets; it neither reads the
dataset text projection nor decodes cached text for any row.  Once a proposal
has frozen and ``fit_predict`` is invoked, only the explicitly supplied context
and gate rows are bound to their cached chunks.  The cache refuses every other
row and cannot launch the configured embedding language model on this host.  The
much smaller supervised HTR encoder remains a normal upstream prediction model
and is fitted only on context rows.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ..config import AppliedInferenceConfig, ExperimentConfig
from ..models.concept_embedding_utils import chunk_text_words
from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .context_fit_upstream_gate_provider import ContextFitUpstreamPrediction
from .context_prediction_htr_provider import (
    CONTEXT_PREDICTION_HTR_PROVIDER_ID,
    HistoricalStage1ContextPredictionHTRProvider,
    context_prediction_htr_provider_identity,
)
from .multi_model_forest_stage1 import (
    MultiModelForestStage1Runner,
)

STAGE1_CONTEXT_BACKEND_ID = "historical_stage1_context_gate_backend_v4"
FROZEN_CHUNK_PROVIDER_ID = "exact_frozen_chunk_embedding_provider_v2"
EFFECTIVE_STAGE1_CONFIG_ID = "historical_stage1_effective_runtime_config_v1"
HTR_RUNTIME_SOURCE_ATTESTATION_ID = "historical_stage1_htr_runtime_sources_v2"

_REQUIRED_CACHE_FILES = (
    "metadata.json",
    "chunk_embeddings.npy",
    "offsets.npy",
    "chunk_texts.jsonl",
)
_DEFAULT_REQUIRED_FAMILIES = frozenset(
    {
        "bow_nuisance",
        "htr_nuisance",
        "bow_weighted_r",
        "htr_weighted_r",
        "htr_neural",
        "matched_pair_uplift",
        "embedding_whole_cohort",
        "embedding_clustered",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _read_stable_bytes(path: Path) -> tuple[bytes, tuple[int, int, int, int, int]]:
    """Read one authenticated snapshot and reject a path change during the read."""

    before = _stat_signature(path)
    payload = path.read_bytes()
    after = _stat_signature(path)
    if before != after:
        raise RuntimeError(f"cache file changed while it was being authenticated: {path.name}")
    return payload, after


def _load_detached_npy(payload: bytes, *, name: str) -> np.ndarray:
    """Parse exactly the bytes that were hashed, never a mutable cache path."""

    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"frozen embedding cache contains an invalid {name} array") from exc
    if not isinstance(loaded, np.ndarray):
        raise ValueError(f"frozen embedding cache {name} must be one NumPy array")
    output = np.array(loaded, copy=True)
    output.setflags(write=False)
    return output


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _effective_applied_config_sha256(config: AppliedInferenceConfig) -> str:
    """Digest the complete live Stage-1 config after intentional overrides.

    The private HTR snapshot lives in a randomly named temporary directory, so
    its path cannot be part of a reusable producer identity.  The backend
    separately requires that live path to equal its authenticated private
    snapshot exactly; here only that one value is replaced by a closed marker.
    Every other effective config field is serialized without omission.
    """

    if type(config) is not AppliedInferenceConfig:
        raise TypeError("effective Stage-1 config must be AppliedInferenceConfig")
    payload = asdict(config)
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict) or "htr_sentence_model" not in architecture:
        raise ValueError("effective Stage-1 config has no HTR sentence-model field")
    architecture["htr_sentence_model"] = {
        "binding": "authenticated_private_htr_model_tree",
    }
    return _canonical_sha256(
        {
            "schema_version": EFFECTIVE_STAGE1_CONFIG_ID,
            "effective_applied_inference_config": payload,
        }
    )


def _htr_runtime_source_attestation() -> Mapping[str, Any]:
    """Return exact source hashes for the production HTR construction path."""

    import oci.inference.agentic_attention_variable_forest as attention_module
    import oci.inference.context_prediction_htr_provider as context_provider_module
    import oci.inference.multi_model_agentic_forest as agentic_module
    import oci.inference.multi_model_forest_stage1 as stage1_module
    import oci.inference.multi_model_pair_uplift as pair_module
    import oci.models.extractor_factory as factory_module
    import oci.models.hierarchical_transformer_extractor as extractor_module
    import oci.utils.calibration as calibration_module

    return {
        "schema_version": HTR_RUNTIME_SOURCE_ATTESTATION_ID,
        "multi_model_forest_stage1_sha256": _module_file_sha256(stage1_module.__file__),
        "multi_model_agentic_forest_sha256": _module_file_sha256(agentic_module.__file__),
        "agentic_attention_variable_forest_sha256": _module_file_sha256(attention_module.__file__),
        "context_prediction_htr_provider_sha256": _module_file_sha256(
            context_provider_module.__file__
        ),
        "multi_model_pair_uplift_sha256": _module_file_sha256(pair_module.__file__),
        "extractor_factory_sha256": _module_file_sha256(factory_module.__file__),
        "hierarchical_transformer_extractor_sha256": _module_file_sha256(extractor_module.__file__),
        "binary_probability_calibration_sha256": _module_file_sha256(calibration_module.__file__),
    }


def _module_file_sha256(module_file: str) -> str:
    return _sha256_file(Path(module_file).resolve())


def _directory_tree_sha256(path: Path) -> str:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": candidate.relative_to(path).as_posix(),
                "size": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    if not rows:
        raise ValueError(f"model directory contains no files: {path}")
    return _canonical_sha256(rows)


def _parse_historical_applied_config(payload: bytes) -> AppliedInferenceConfig:
    try:
        saved = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("historical Stage-1 config is not valid UTF-8 JSON") from exc
    if not isinstance(saved, Mapping):
        raise ValueError("historical Stage-1 config must be one JSON object")
    config_payload = saved.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("historical Stage-1 config has no applied config payload")
    applied = copy.deepcopy(dict(config_payload))
    architecture = applied.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("historical Stage-1 config has no architecture payload")
    architecture = copy.deepcopy(dict(architecture))
    # This duplicate legacy object contains keys that no longer belong to the
    # base dataclass. The integrated multi_model_forest payload is the actual
    # builder configuration and remains fully parsed below.
    architecture.pop("multi_model_agentic_forest", None)
    if not isinstance(architecture.get("multi_model_forest"), Mapping):
        raise ValueError("historical Stage-1 config has no multi_model_forest payload")
    applied["architecture"] = architecture
    return ExperimentConfig.from_dict({"applied_inference": applied}).applied_inference


@dataclass(frozen=True)
class HistoricalStage1ConfigSnapshot:
    """One exact config byte snapshot shared by every adaptive-review consumer."""

    source_path: Path
    sha256: str
    _source_stat: tuple[int, int, int, int, int]
    _payload: bytes

    @classmethod
    def from_path(cls, stage1_config_path: Path | str) -> "HistoricalStage1ConfigSnapshot":
        path = Path(stage1_config_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"historical Stage-1 config does not exist: {path}")
        payload, source_stat = _read_stable_bytes(path)
        _parse_historical_applied_config(payload)
        digest = hashlib.sha256(payload).hexdigest()
        # Parsing never consults the path, but reject a replacement that lands
        # between the byte read and completion of the parsed snapshot object.
        before = _stat_signature(path)
        current_digest = _sha256_file(path)
        after = _stat_signature(path)
        if before != source_stat or after != before or current_digest != digest:
            raise RuntimeError("historical Stage-1 config changed during snapshotting")
        return cls(
            source_path=path,
            sha256=digest,
            _source_stat=source_stat,
            _payload=payload,
        )

    def applied_config(self) -> AppliedInferenceConfig:
        return _parse_historical_applied_config(self._payload)

    def verify_source(self) -> None:
        try:
            before = _stat_signature(self.source_path)
            digest = _sha256_file(self.source_path)
            after = _stat_signature(self.source_path)
        except OSError as exc:
            raise RuntimeError("historical Stage-1 config path changed after snapshotting") from exc
        if before != self._source_stat or after != before or digest != self.sha256:
            raise RuntimeError("historical Stage-1 config path changed after snapshotting")


def _historical_stage1_config_snapshot(
    stage1_config_path: Path | str | None,
    snapshot: HistoricalStage1ConfigSnapshot | None = None,
) -> HistoricalStage1ConfigSnapshot:
    if snapshot is not None:
        if not isinstance(snapshot, HistoricalStage1ConfigSnapshot):
            raise TypeError("stage1_config_snapshot must be HistoricalStage1ConfigSnapshot")
        if stage1_config_path is not None:
            requested = Path(stage1_config_path).resolve()
            if requested != snapshot.source_path:
                raise ValueError("Stage-1 config path does not match supplied exact snapshot")
        snapshot.verify_source()
        return snapshot
    if stage1_config_path is None:
        raise ValueError("stage1_config_path or stage1_config_snapshot is required")
    return HistoricalStage1ConfigSnapshot.from_path(stage1_config_path)


class PrivateHTRModelTreeSnapshot:
    """Retain a byte-exact, read-only private copy of one HTR model tree."""

    def __init__(self, source_path: Path | str) -> None:
        source = Path(source_path).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"HTR model directory does not exist: {source}")
        candidates = sorted(item for item in source.rglob("*") if item.is_file())
        if not candidates:
            raise ValueError(f"HTR model directory contains no files: {source}")
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="oci-htr-snapshot-")
        private_root = Path(self._temporary_directory.name) / source.name
        private_root.mkdir(parents=True, exist_ok=False)
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            relative = candidate.relative_to(source)
            destination = private_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            before = _stat_signature(candidate)
            digest = hashlib.sha256()
            size = 0
            with candidate.open("rb") as input_handle, destination.open("wb") as output_handle:
                for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    output_handle.write(chunk)
                    size += len(chunk)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            after = _stat_signature(candidate)
            if before != after:
                raise RuntimeError(f"HTR model file changed during snapshotting: {relative}")
            rows.append(
                {
                    "relative_path": relative.as_posix(),
                    "size": size,
                    "sha256": digest.hexdigest(),
                }
            )
        snapshot_digest = _canonical_sha256(rows)
        if _directory_tree_sha256(private_root) != snapshot_digest:
            raise RuntimeError("private HTR model snapshot failed byte verification")
        if _directory_tree_sha256(source) != snapshot_digest:
            raise RuntimeError("source HTR model tree changed during snapshotting")
        for candidate in private_root.rglob("*"):
            if candidate.is_file():
                candidate.chmod(0o400)
        for directory in sorted(
            (item for item in private_root.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            directory.chmod(0o500)
        private_root.chmod(0o500)
        self.path = private_root
        self.sha256 = snapshot_digest
        self.source_path = source
        self.source_basename = source.name

    def verify(self) -> None:
        if _directory_tree_sha256(self.path) != self.sha256:
            raise RuntimeError("private HTR model snapshot changed after authentication")


class ExactFrozenChunkEmbeddingProvider:
    """Serve one existing chunk cache and reject all novel encoding requests."""

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        dataset_texts: Sequence[str],
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        for filename in _REQUIRED_CACHE_FILES:
            path = self.cache_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"incomplete frozen chunk cache: {path}")
        snapshots: dict[str, bytes] = {}
        file_stats: dict[str, tuple[int, int, int, int, int]] = {}
        for filename in _REQUIRED_CACHE_FILES:
            snapshots[filename], file_stats[filename] = _read_stable_bytes(
                self.cache_dir / filename
            )
        # A file read early in the sequence must also remain the authenticated
        # path through completion of the full four-file snapshot.
        for filename, signature in file_stats.items():
            if _stat_signature(self.cache_dir / filename) != signature:
                raise RuntimeError(
                    f"cache file changed while it was being authenticated: {filename}"
                )
        try:
            metadata = json.loads(snapshots["metadata.json"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("frozen embedding cache metadata is invalid JSON") from exc
        if not isinstance(metadata, dict):
            raise ValueError("frozen embedding cache metadata must be an object")
        self._metadata = metadata
        self._embeddings = _load_detached_npy(
            snapshots["chunk_embeddings.npy"], name="chunk_embeddings"
        )
        self._offsets = _load_detached_npy(snapshots["offsets.npy"], name="offsets")
        exact_texts = tuple(str(value) for value in dataset_texts)
        if int(self._metadata.get("num_samples", -1)) != len(exact_texts):
            raise ValueError("frozen chunk cache row count does not match text projection")
        if self._offsets.ndim != 1 or len(self._offsets) != len(exact_texts) + 1:
            raise ValueError("frozen chunk cache offsets do not match text projection")
        if self._embeddings.ndim != 2:
            raise ValueError("frozen chunk cache embedding matrix must be two-dimensional")
        if int(self._offsets[-1]) != int(self._embeddings.shape[0]):
            raise ValueError("frozen chunk cache offsets do not span its embedding matrix")
        if int(self._metadata.get("hidden_size", -1)) != int(self._embeddings.shape[1]):
            raise ValueError("frozen chunk cache hidden size is inconsistent")

        cached_rows: list[tuple[str, ...]] = []
        for line in snapshots["chunk_texts.jsonl"].splitlines():
            try:
                payload = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("frozen chunk text registry contains invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("frozen chunk text registry row must be an object")
            chunks = payload.get("chunks")
            if not isinstance(chunks, list) or not all(isinstance(value, str) for value in chunks):
                raise ValueError("frozen chunk text registry row has an invalid schema")
            cached_rows.append(tuple(chunks))
        if len(cached_rows) != len(exact_texts):
            raise ValueError("frozen chunk text rows do not match cache metadata")
        generated_rows = tuple(
            tuple(
                chunk_text_words(
                    text,
                    int(self._metadata["chunk_size_words"]),
                    int(self._metadata["chunk_overlap_words"]),
                    int(self._metadata["max_chunks"]),
                    "last",
                )
            )
            for text in exact_texts
        )
        cached_flat_chunks = tuple(chunk for row in cached_rows for chunk in row)
        if len(cached_flat_chunks) != len(self._embeddings):
            raise ValueError("frozen chunk text count does not match embedding rows")
        token_bounded_rows: list[int] = []
        for row_id, (cached, generated) in enumerate(zip(cached_rows, generated_rows)):
            if cached == generated:
                continue
            # A few very long rows were split by the cache writer's secondary
            # tokenizer bound. It left soft-hyphen padding at the front of
            # those chunks. Bind them to the same source text by requiring the
            # unchanged final word-chunk and a real cached prefix found in the
            # source document. Ordinary text changes still fail closed.
            if not generated or not cached or cached[-1] != generated[-1]:
                raise ValueError(
                    "frozen chunk cache texts are not the exact configured dataset " "projection"
                )
            first_real = next(
                (
                    chunk.replace("\u00ad", "").strip()
                    for chunk in cached
                    if chunk.replace("\u00ad", "").strip()
                ),
                "",
            )
            normalized_source = " ".join(exact_texts[row_id].split())
            normalized_first = " ".join(first_real.split())
            if not normalized_first or normalized_first not in normalized_source:
                raise ValueError("token-bounded cache row cannot be bound to dataset text")
            token_bounded_rows.append(row_id)
        if len(token_bounded_rows) > max(4, int(math.ceil(0.01 * len(exact_texts)))):
            raise ValueError("too many token-bounded cache rows for exact audit")
        self._cached_rows = tuple(cached_rows)
        self._token_bounded_rows = tuple(token_bounded_rows)
        self._flat_chunks = cached_flat_chunks
        self._identity = {
            "provider": FROZEN_CHUNK_PROVIDER_ID,
            "metadata_sha256": hashlib.sha256(snapshots["metadata.json"]).hexdigest(),
            "embeddings_sha256": hashlib.sha256(snapshots["chunk_embeddings.npy"]).hexdigest(),
            "offsets_sha256": hashlib.sha256(snapshots["offsets.npy"]).hexdigest(),
            "chunk_texts_sha256": hashlib.sha256(snapshots["chunk_texts.jsonl"]).hexdigest(),
            "row_count": len(exact_texts),
            "chunk_count": len(cached_flat_chunks),
            "token_bounded_row_count": len(self._token_bounded_rows),
            "token_bounded_row_ids_sha256": _canonical_sha256(list(self._token_bounded_rows)),
            "cache_snapshot_authentication": "single_read_sha256_detached_numpy_v1",
            "embeddings_path_backed": False,
            "novel_text_encoding_allowed": False,
        }
        self._file_stats = file_stats

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return metadata detached from the authenticated internal contract."""

        return copy.deepcopy(self._metadata)

    def identity(self) -> Mapping[str, Any]:
        hash_fields = {
            "metadata.json": "metadata_sha256",
            "chunk_embeddings.npy": "embeddings_sha256",
            "offsets.npy": "offsets_sha256",
            "chunk_texts.jsonl": "chunk_texts_sha256",
        }
        for filename, field_name in hash_fields.items():
            path = self.cache_dir / filename
            try:
                before = _stat_signature(path)
                digest = _sha256_file(path)
                after = _stat_signature(path)
            except OSError as exc:
                raise RuntimeError(
                    f"frozen embedding cache path changed after authentication: {filename}"
                ) from exc
            if (
                before != self._file_stats[filename]
                or after != before
                or digest != self._identity[field_name]
            ):
                raise RuntimeError(
                    f"frozen embedding cache path changed after authentication: {filename}"
                )
        return copy.deepcopy(self._identity)

    def encode_chunks(self, values: Sequence[str]) -> np.ndarray:
        requested = tuple(str(value) for value in values)
        if requested != self._flat_chunks:
            raise ValueError(
                "frozen embedding provider refuses novel, reordered, or partial chunks"
            )
        return np.array(self._embeddings, dtype=np.float32, copy=True)

    def chunk_matrix(self, row_id: int) -> np.ndarray:
        if not 0 <= int(row_id) < len(self._offsets) - 1:
            raise IndexError("frozen embedding row ID is out of range")
        start = int(self._offsets[int(row_id)])
        stop = int(self._offsets[int(row_id) + 1])
        return np.array(self._embeddings[start:stop], dtype=np.float32, copy=True)

    def chunk_matrices(self, row_ids: Sequence[int]) -> list[np.ndarray]:
        """Return immutable-cache matrices in the requested canonical row order."""

        return [self.chunk_matrix(int(row_id)) for row_id in row_ids]

    def chunk_texts(self, row_ids: Sequence[int]) -> list[list[str]]:
        """Return cached chunk text only for explicitly requested rows.

        Keeping the row selection at this boundary makes it straightforward for
        spent-only discovery code to avoid materializing text from sealed review
        partitions even though the underlying representation cache spans the
        complete dataset.
        """

        output: list[list[str]] = []
        for raw_row_id in row_ids:
            row_id = int(raw_row_id)
            if not 0 <= row_id < len(self._cached_rows):
                raise IndexError("frozen embedding row ID is out of range")
            output.append(list(self._cached_rows[row_id]))
        return output


class _FrozenEmbeddingGenerator:
    """Minimal Stage-1 generator facade backed only by frozen cache bytes."""

    def __init__(self, provider: ExactFrozenChunkEmbeddingProvider) -> None:
        self.provider = provider

    def prepare(self, dataset: pd.DataFrame) -> None:
        if len(dataset) != int(self.provider.metadata["num_samples"]):
            raise ValueError("Stage-1 dataset rows do not match frozen embedding cache")

    @staticmethod
    def _positions_for_frame(frame: pd.DataFrame) -> list[int]:
        if "_oci_row_id" not in frame.columns:
            raise ValueError("embedding frames require canonical row IDs")
        return [int(value) for value in frame["_oci_row_id"]]

    def _chunk_matrix(self, position: int) -> np.ndarray:
        return self.provider.chunk_matrix(int(position))

    def _patient_embeddings(self, positions: Sequence[int]) -> np.ndarray:
        rows: list[np.ndarray] = []
        hidden_size = int(self.provider.metadata["hidden_size"])
        for position in positions:
            chunks = self._chunk_matrix(int(position))
            rows.append(
                np.mean(chunks, axis=0) if len(chunks) else np.zeros(hidden_size, dtype=np.float32)
            )
        return np.vstack(rows).astype(np.float32, copy=False)

    def build_evidence(self, **_kwargs: Any) -> dict[str, Any]:
        # Numerical gate evaluation does not need retrieved excerpts or concept
        # probes. Suppressing them prevents every novel embedding request.
        return {}


def _minimal_historical_applied_config(stage1_config_path: Path) -> AppliedInferenceConfig:
    return HistoricalStage1ConfigSnapshot.from_path(stage1_config_path).applied_config()


def _resolve_htr_model_path(config: AppliedInferenceConfig) -> Path:
    raw = Path(str(config.architecture.htr_sentence_model)).expanduser()
    candidates = [raw]
    raw_text = str(raw)
    if raw_text.startswith("/homes/"):
        candidates.append(Path("/home") / Path(raw_text).relative_to("/homes"))
    for candidate in candidates:
        if candidate.is_dir():
            config.architecture.htr_sentence_model = str(candidate.resolve())
            return candidate.resolve()
    raise FileNotFoundError(
        "the historical HTR sentence encoder is unavailable locally: " f"{raw_text}"
    )


def _spent_only_embedding_cache(
    cache_dir: Path | str | None,
    embedding_cache: Any | None = None,
) -> Any:
    """Construct the public lazy cache without creating an import cycle.

    ``review_spent_evidence_provider`` imports the Stage-1 helpers in this
    module, so importing its public cache at module import time would be a
    cycle.  Backend construction happens only after both modules have loaded.
    The returned cache scans raw JSONL byte offsets but decodes semantic chunk
    text only when ``bind_spent`` is called for explicitly supplied rows.
    """

    from .review_spent_evidence_provider import SpentOnlyFrozenChunkEmbeddingCache

    if embedding_cache is not None:
        if not isinstance(embedding_cache, SpentOnlyFrozenChunkEmbeddingCache):
            raise TypeError("embedding_cache must be SpentOnlyFrozenChunkEmbeddingCache")
        if cache_dir is not None and embedding_cache.cache_dir != Path(cache_dir).resolve():
            raise ValueError("embedding_cache_dir does not match supplied embedding_cache")
        return embedding_cache
    if cache_dir is None:
        raise ValueError("embedding_cache_dir or embedding_cache is required")
    return SpentOnlyFrozenChunkEmbeddingCache(cache_dir)


class HistoricalStage1ContextBackend:
    """Regenerate all legacy Stage-1 sources on a spent review context."""

    def __init__(
        self,
        *,
        dataset_path: Path | str,
        stage1_config_path: Path | str | None = None,
        embedding_cache_dir: Path | str | None = None,
        stage1_config_snapshot: HistoricalStage1ConfigSnapshot | None = None,
        embedding_cache: Any | None = None,
        htr_model_snapshot: PrivateHTRModelTreeSnapshot | None = None,
        device: str = "cuda:0",
        bow_fold_parallelism: int = 1,
        bow_parallel_backend: str = "threads",
        required_families: Sequence[str] = tuple(sorted(_DEFAULT_REQUIRED_FAMILIES)),
    ) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        if not self.dataset_path.is_file():
            raise FileNotFoundError("historical Stage-1 dataset must exist")
        self._stage1_config_snapshot = _historical_stage1_config_snapshot(
            stage1_config_path,
            stage1_config_snapshot,
        )
        self.stage1_config_path = self._stage1_config_snapshot.source_path
        self.config = self._stage1_config_snapshot.applied_config()
        self.config.dataset_path = str(self.dataset_path)
        if isinstance(bow_fold_parallelism, (bool, np.bool_)) or not isinstance(
            bow_fold_parallelism, (int, np.integer)
        ):
            raise TypeError("bow_fold_parallelism must be an integer")
        self.bow_fold_parallelism = int(bow_fold_parallelism)
        if self.bow_fold_parallelism < 1:
            raise ValueError("bow_fold_parallelism must be positive")
        self.bow_parallel_backend = str(bow_parallel_backend).strip().lower()
        if self.bow_parallel_backend == "loky":
            self.bow_parallel_backend = "processes"
        if self.bow_parallel_backend not in {"threads", "processes"}:
            raise ValueError("bow_parallel_backend must be 'threads' or 'processes'")
        self.config.architecture.multi_model_forest.outer_parallelism = "1"
        self.config.architecture.multi_model_forest.fold_parallelism = "1"
        self.config.architecture.multi_model_forest.bow_fold_parallelism = str(
            self.bow_fold_parallelism
        )
        self.config.architecture.multi_model_forest.htr_fold_parallelism = "1"
        self.config.architecture.multi_model_forest.cpus_total = self.bow_fold_parallelism
        self.config.architecture.multi_model_forest.bow_parallel_backend = (
            self.bow_parallel_backend
        )
        self.config.architecture.agentic_attention_variable_forest.fold_parallelism = "1"
        self.config.architecture.htr_require_live_unfrozen_encoder_attestation = True
        embedding_config = self.config.architecture.multi_model_forest.embedding_contrast
        embedding_config.include_bow_phrases_as_concepts = False
        embedding_config.concept_phrases = []
        embedding_config.external_corpus_cache_dirs = []
        htr_path = _resolve_htr_model_path(self.config)
        if htr_model_snapshot is not None:
            if not isinstance(htr_model_snapshot, PrivateHTRModelTreeSnapshot):
                raise TypeError("htr_model_snapshot must be PrivateHTRModelTreeSnapshot")
            if htr_model_snapshot.source_path != htr_path:
                raise ValueError("HTR model path does not match supplied private snapshot")
            htr_model_snapshot.verify()
            self._htr_model_snapshot = htr_model_snapshot
        else:
            self._htr_model_snapshot = PrivateHTRModelTreeSnapshot(htr_path)
        self.config.architecture.htr_sentence_model = str(self._htr_model_snapshot.path)
        self.device = str(device)
        if not self.device.startswith("cuda:") and self.device != "cpu":
            raise ValueError("device must be 'cpu' or one explicit CUDA device")

        # The embedding cache supplies the canonical global row count.  Keep a
        # blank positional frame for legacy Stage-1 internals, but do not read
        # or retain any dataset text at construction time.  Semantic cache text
        # is decoded later only for the context/gate rows explicitly supplied
        # to fit_predict().
        self.embedding_cache = _spent_only_embedding_cache(
            embedding_cache_dir,
            embedding_cache,
        )
        text_column = str(self.config.text_column)
        self._dataset_frame = pd.DataFrame(
            {
                "_oci_row_id": np.arange(self.embedding_cache.row_count, dtype=int),
                text_column: [""] * self.embedding_cache.row_count,
            }
        )
        self.required_families = frozenset(
            str(value).strip() for value in required_families if str(value).strip()
        )
        if not self.required_families:
            raise ValueError("required_families cannot be empty")

        import oci.inference.embedding_contrast_discovery as embedding_module
        import oci.inference.multi_model_forest_stage1 as stage1_module
        import oci.inference.multi_model_pair_uplift as pair_module

        effective_config_sha256 = self.effective_config_sha256()
        htr_runtime_sources = self.htr_runtime_source_attestation()
        context_htr_provider_identity = context_prediction_htr_provider_identity(
            self.config,
            device=self.device,
        )

        self._identity = {
            "backend": STAGE1_CONTEXT_BACKEND_ID,
            "stage1_config_sha256": self._stage1_config_snapshot.sha256,
            "effective_config_schema_version": EFFECTIVE_STAGE1_CONFIG_ID,
            "effective_config_sha256": effective_config_sha256,
            "dataset_row_count": self.embedding_cache.row_count,
            "embedding_provider": self.embedding_cache.identity(),
            "htr_model_tree_sha256": self._htr_model_snapshot.sha256,
            "htr_model_path_basename": self._htr_model_snapshot.source_basename,
            "htr_model_source_path_used_after_snapshot": False,
            "htr_runtime_source_attestation": htr_runtime_sources,
            "context_prediction_htr_provider": context_htr_provider_identity,
            "context_prediction_htr_provider_required": True,
            "context_prediction_htr_provider_id": CONTEXT_PREDICTION_HTR_PROVIDER_ID,
            "stage1_code_sha256": _module_file_sha256(stage1_module.__file__),
            "pair_code_sha256": _module_file_sha256(pair_module.__file__),
            "embedding_code_sha256": _module_file_sha256(embedding_module.__file__),
            "device": self.device,
            "bow_fold_parallelism": self.bow_fold_parallelism,
            "bow_parallel_backend": self.bow_parallel_backend,
            "htr_fold_parallelism": 1,
            "required_families": sorted(self.required_families),
            "embedding_language_model_launch_allowed": False,
            "gate_labels_exposed": False,
            "context_train_pair_or_effect_predictions_consumed": False,
            "spent_discovery_path_changed": False,
            "dataset_text_read_or_hashed_at_construction": False,
            "future_row_text_decoded_or_materialized": False,
            "context_gate_text_binding": "fit_predict_explicit_rows_only_after_proposal",
        }

    def effective_config_sha256(self) -> str:
        """Recompute the closed digest of the exact live effective config."""

        configured_model = Path(str(self.config.architecture.htr_sentence_model)).resolve()
        if configured_model != self._htr_model_snapshot.path.resolve():
            raise RuntimeError(
                "effective Stage-1 config no longer points at its private HTR snapshot"
            )
        return _effective_applied_config_sha256(self.config)

    def htr_runtime_source_attestation(self) -> Mapping[str, Any]:
        """Recompute source hashes for the complete production HTR runtime path."""

        return _htr_runtime_source_attestation()

    def identity(self) -> Mapping[str, Any]:
        self._stage1_config_snapshot.verify_source()
        self._htr_model_snapshot.verify()
        if self.effective_config_sha256() != self._identity["effective_config_sha256"]:
            raise RuntimeError("effective Stage-1 runtime config changed after construction")
        if (
            self.htr_runtime_source_attestation()
            != self._identity["htr_runtime_source_attestation"]
        ):
            raise RuntimeError("production HTR runtime source changed after construction")
        if (
            context_prediction_htr_provider_identity(self.config, device=self.device)
            != self._identity["context_prediction_htr_provider"]
        ):
            raise RuntimeError("context-prediction HTR provider identity changed")
        current_cache_identity = self.embedding_cache.identity()
        if current_cache_identity != self._identity["embedding_provider"]:
            raise RuntimeError("Stage-1 context embedding cache identity changed")
        return copy.deepcopy(self._identity)

    @staticmethod
    def _feature_kind(metadata: Mapping[str, Any]) -> str:
        family = str(metadata.get("source_family") or "").strip().lower()
        objective = str(metadata.get("objective") or "").strip().lower()
        contrast_family = str(metadata.get("contrast_family") or "").strip().lower()
        if family in {"bow_pair_uplift", "htr_pair_uplift"}:
            return "matched_pair_uplift"
        if family == "embedding_contrast":
            if "cluster" in contrast_family or "cluster" in objective:
                return "embedding_clustered"
            return "embedding_whole_cohort"
        if family == "htr":
            return "htr_nuisance" if "nuisance" in objective else "htr_neural"
        if family == "bow":
            return "bow_nuisance" if "nuisance" in objective else "bow_r_loss"
        return family or "stage1_upstream"

    @staticmethod
    def _w_roles(metadata: Mapping[str, Any]) -> tuple[str, ...]:
        objective = str(metadata.get("objective") or "").lower()
        treatment = "treatment" in objective
        outcome = "outcome" in objective
        if treatment and outcome:
            return (
                PROPENSITY_NUISANCE_FEATURE_ROLE,
                OUTCOME_NUISANCE_FEATURE_ROLE,
            )
        if treatment:
            return (PROPENSITY_NUISANCE_FEATURE_ROLE,)
        if outcome:
            return (OUTCOME_NUISANCE_FEATURE_ROLE,)
        # W columns are adjustment bases.  When their historical name does not
        # distinguish a nuisance target, preserve against both nuisance fits.
        return (
            PROPENSITY_NUISANCE_FEATURE_ROLE,
            OUTCOME_NUISANCE_FEATURE_ROLE,
        )

    def _observed_families(
        self,
        source_names: Sequence[str],
        feature_kinds: Sequence[str],
    ) -> frozenset[str]:
        observed = set(feature_kinds)
        for name in source_names:
            lowered = name.lower()
            if "bow" in lowered:
                observed.add("bow_weighted_r")
            if "htr" in lowered:
                observed.add("htr_weighted_r")
        return frozenset(observed)

    def fit_predict(
        self,
        *,
        outer_fold: int,
        context_row_ids: tuple[int, ...],
        context_texts: tuple[str, ...],
        context_treatment: np.ndarray,
        context_outcome: np.ndarray,
        gate_row_ids: tuple[int, ...],
        gate_texts: tuple[str, ...],
        work_dir: Path,
    ) -> ContextFitUpstreamPrediction:
        if set(context_row_ids) & set(gate_row_ids):
            raise ValueError("Stage-1 context and gate must be disjoint")
        self.identity()
        # This is the first semantic access to cached text.  The runner invokes
        # this backend only after a proposal has been validated/applied and its
        # extraction-quality guard has passed.  The bound provider authenticates
        # each exact supplied row/text pair and refuses all unbound rows.
        embedding_provider = self.embedding_cache.bind_spent(
            tuple(context_row_ids) + tuple(gate_row_ids),
            tuple(context_texts) + tuple(gate_texts),
        )
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(self.config)
        gpu_ids = (int(self.device.split(":", 1)[1]),) if self.device.startswith("cuda:") else None
        htr_provider = HistoricalStage1ContextPredictionHTRProvider(
            config=config,
            output_dir=work_dir / "stage1_text_models",
            device=torch.device(self.device),
            gpu_ids=gpu_ids,
            num_workers=1,
        )
        if htr_provider.identity() != self._identity["context_prediction_htr_provider"]:
            raise RuntimeError("constructed context-prediction HTR provider is inexact")
        runner = MultiModelForestStage1Runner(
            dataset=self._dataset_frame,
            config=config,
            output_path=work_dir / "unused_predictions.parquet",
            device=torch.device(self.device),
            gpu_ids=gpu_ids,
            num_workers=1,
            embedding_provider=None,
            htr_evidence_provider=htr_provider,
        )
        runner.embedding_evidence_generator = _FrozenEmbeddingGenerator(embedding_provider)
        train_df = pd.DataFrame(
            {
                "_oci_row_id": context_row_ids,
                config.text_column: context_texts,
                config.treatment_column: np.asarray(context_treatment, dtype=float),
                config.outcome_column: np.asarray(context_outcome, dtype=float),
            }
        )
        # Intentionally label-free: the Stage-1 builder has no gate treatment,
        # outcome, or synthetic benchmark field available to read.
        test_df = pd.DataFrame(
            {
                "_oci_row_id": gate_row_ids,
                config.text_column: gate_texts,
            }
        )
        self._htr_model_snapshot.verify()
        try:
            bundle = runner._build_feature_bundle(
                train_df=train_df,
                test_df=test_df,
                outer_fold=int(outer_fold),
            )
            prediction_bundle = htr_provider.seal_prediction_only_bundle(bundle)
        finally:
            self._htr_model_snapshot.verify()
        x_test = prediction_bundle.x_test
        w_test = prediction_bundle.w_test
        metadata = {str(row["feature_name"]): row for row in prediction_bundle.feature_rows}
        if set(prediction_bundle.x_names) | set(prediction_bundle.w_names) != set(metadata):
            raise ValueError("Stage-1 feature metadata does not match matrix columns")

        source_names: list[str] = []
        source_kinds: list[str] = []
        source_columns: list[np.ndarray] = []
        feature_names: list[str] = []
        feature_kinds: list[str] = []
        feature_roles: list[str] = []
        feature_columns: list[np.ndarray] = []

        for column, name in enumerate(prediction_bundle.x_names):
            row = metadata[name]
            family = str(row.get("source_family") or "").lower()
            objective = str(row.get("objective") or "").lower()
            if objective == "direct_weighted_r" and family in {"bow", "htr"}:
                source_names.append(f"stage1_calibrated__{name}")
                source_kinds.append(f"nested_calibrated_{family}_weighted_r")
                source_columns.append(np.asarray(x_test[:, column], dtype=float))
                continue
            feature_names.append(f"stage1_raw__{name}")
            feature_kinds.append(self._feature_kind(row))
            feature_roles.append(UNCALIBRATED_EFFECT_MODIFIER_ROLE)
            feature_columns.append(np.asarray(x_test[:, column], dtype=float))

        for column, name in enumerate(prediction_bundle.w_names):
            row = metadata[name]
            roles = self._w_roles(row)
            for role in roles:
                suffix = "propensity" if role == PROPENSITY_NUISANCE_FEATURE_ROLE else "outcome"
                feature_names.append(f"stage1_raw__{name}__as_{suffix}")
                feature_kinds.append(self._feature_kind(row))
                feature_roles.append(role)
                feature_columns.append(np.asarray(w_test[:, column], dtype=float))

        observed = self._observed_families(source_names, feature_kinds)
        missing = sorted(self.required_families - observed)
        if missing:
            raise RuntimeError(
                "context-fitted Stage-1 bundle is missing required upstream families: "
                + ", ".join(missing)
            )
        self.identity()
        return ContextFitUpstreamPrediction(
            gate_row_ids=gate_row_ids,
            calibrated_source_names=tuple(source_names),
            calibrated_source_kinds=tuple(source_kinds),
            calibrated_source_values=np.column_stack(source_columns),
            feature_names=tuple(feature_names),
            feature_kinds=tuple(feature_kinds),
            feature_roles=tuple(feature_roles),
            feature_values=np.column_stack(feature_columns),
        )


__all__ = [
    "FROZEN_CHUNK_PROVIDER_ID",
    "STAGE1_CONTEXT_BACKEND_ID",
    "ExactFrozenChunkEmbeddingProvider",
    "HistoricalStage1ConfigSnapshot",
    "HistoricalStage1ContextBackend",
    "PrivateHTRModelTreeSnapshot",
]
